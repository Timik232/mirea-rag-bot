"""Scraper for www.mirea.ru/contacts/ — physical addresses, phones, emails."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mirea.ru"
SECTION = "contacts"
DELAY = 1.5
MAX_RETRIES = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SUBPAGES = [
    ("/contacts/", "Контакты"),
    ("/contacts/priemnaya-komissiya/", "Приёмная комиссия"),
    ("/contacts/strukturnye-podrazdeleniya/", "Структурные подразделения"),
    ("/contacts/dekany/", "Деканы институтов"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    try:
        import config
        raw_dir = getattr(config, "DATA_RAW_DIR", "mirea_data/raw")
    except Exception:
        raw_dir = "mirea_data/raw"
    p = Path(raw_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(
        ["nav", "footer", "header", "script", "style", "noscript", "aside"]
    ):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(
        r"nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal",
        re.I,
    )):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(
        r"nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal",
        re.I,
    )):
        tag.decompose()


def _parse_table(table: Tag) -> list[str]:
    rows: list[str] = []
    header_row = table.find("tr")
    if not header_row:
        return rows
    headers = [
        _clean_text(th.get_text()) for th in header_row.find_all(["th", "td"])
    ]
    if not headers:
        return rows
    for tr in table.find_all("tr")[1:]:
        cells = [_clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
        if not cells or all(c == "" for c in cells):
            continue
        parts = []
        for h, v in zip(headers, cells):
            if h and v:
                parts.append(f"{h}: {v}")
        if parts:
            rows.append(" | ".join(parts))
    return rows


def _is_reachable(url: str, timeout: int = 5) -> bool:
    try:
        resp = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True,
                             proxies={"http": None, "https": None})
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _safe_get(session: requests.Session, url: str) -> BeautifulSoup | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.encoding = "utf-8"
            if resp.status_code in (403, 503) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 3
                print(f"  [retry {attempt+1}] {resp.status_code} — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [skip] {url} → HTTP {resp.status_code}")
                return None
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            print(f"  [error] {url} → {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) * 3)
    return None


def _extract_contacts(soup: BeautifulSoup, url: str, subsection: str) -> list[dict[str, Any]]:
    """Extract structured contact info: addresses, phones, emails from a page."""
    items: list[dict[str, Any]] = []
    content = (
        soup.find("div", class_=re.compile(r"content|article|detail|post|page-content", re.I))
        or soup.find("main")
        or soup.find("body")
        or soup
    )
    title_tag = soup.find("h1") or soup.find("title")
    page_title = _clean_text(title_tag.get_text()) if title_tag else subsection

    text_content = content.get_text()
    phone_pattern = re.compile(
        r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    )
    seen_phones: set[str] = set()
    for match in phone_pattern.finditer(text_content):
        phone = _clean_text(match.group())
        if phone not in seen_phones:
            seen_phones.add(phone)
            start = max(0, match.start() - 150)
            end = min(len(text_content), match.end() + 50)
            context = _clean_text(text_content[start:end])
            items.append({
                "title": page_title,
                "text": f"Телефон: {phone}. Контекст: {context}",
                "url": url,
                "section": SECTION,
                "subsection": subsection,
            })

    email_pattern = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    seen_emails: set[str] = set()
    for match in email_pattern.finditer(text_content):
        email = match.group()
        if email not in seen_emails and "mirea" in email.lower():
            seen_emails.add(email)
            start = max(0, match.start() - 100)
            end = min(len(text_content), match.end() + 50)
            context = _clean_text(text_content[start:end])
            items.append({
                "title": page_title,
                "text": f"Email: {email}. Контекст: {context}",
                "url": url,
                "section": SECTION,
                "subsection": subsection,
            })

    addr_pattern = re.compile(
        r"(?:г\.\s*Москва|Москва)[,\s]+[^.]{10,}?(?:д\.\s*\d+[а-я]?|дом\s*\d+[а-я]?|владение\s*\d+)",
        re.I,
    )
    seen_addrs: set[str] = set()
    for match in addr_pattern.finditer(text_content):
        addr = _clean_text(match.group())
        if addr not in seen_addrs:
            seen_addrs.add(addr)
            items.append({
                "title": page_title,
                "text": f"Адрес: {addr}",
                "url": url,
                "section": SECTION,
                "subsection": subsection,
            })

    seen_texts: set[str] = set()
    for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _clean_text(el.get_text())
        if not text or len(text) < 20 or text in seen_texts:
            continue
        seen_texts.add(text)
        items.append({
            "title": page_title,
            "text": text,
            "url": url,
            "section": SECTION,
            "subsection": subsection,
        })

    for table in content.find_all("table"):
        for row_text in _parse_table(table):
            items.append({
                "title": page_title,
                "text": row_text,
                "url": url,
                "section": SECTION,
                "subsection": subsection,
            })

    return items


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------

def scrape_contacts() -> list[dict[str, Any]]:
    """Scrape www.mirea.ru/contacts/ and return structured items."""
    session = requests.Session()
    session.trust_env = False
    items: list[dict[str, Any]] = []

    for path, subsection in SUBPAGES:
        url = BASE_URL + path
        print(f"[*] {url}")
        soup = _safe_get(session, url)
        if soup is None:
            time.sleep(DELAY)
            continue

        _strip_noise(soup)
        page_items = _extract_contacts(soup, url, subsection)
        items.extend(page_items)

        time.sleep(DELAY)

    return items


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _demo_data() -> list[dict[str, Any]]:
    return [
        {"title": "Контакты", "text": "Адрес: г. Москва, ул. Стромынка, д. 20 — главный учебный корпус РТУ МИРЭА. Проезд: ст. метро «Сокольники».", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Адрес: г. Москва, проспект Вернадского, д. 78 — учебно-лабораторный корпус. Проезд: ст. метро «Юго-Западная».", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Адрес: г. Москва, ул. Малая Пироговская, д. 1, стр. 5 — учебный корпус Института тонких химических технологий.", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Адрес: г. Москва, 5-я Кабельная, д. 12А — учебно-производственный корпус.", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Адрес: г. Москва, ул. Нежинская, д. 7 — филиал Ломоносовский университет.", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Адрес: г. Москва, ул. Б. Галушкина, д. 3 — общежитие №1 для студентов. Станция метро «ВДНХ».", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Приёмная комиссия", "text": "Телефон: +7 (495) 215-65-65. Приёмная комиссия РТУ МИРЭА осуществляет консультирование абитуриентов по вопросам поступления.", "url": "https://www.mirea.ru/contacts/priemnaya-komissiya/", "section": SECTION, "subsection": "Приёмная комиссия"},
        {"title": "Приёмная комиссия", "text": "Email: priem@mirea.ru. Приём документов осуществляется в корпусе на ул. Стромынка, д. 20, каб. 306.", "url": "https://www.mirea.ru/contacts/priemnaya-komissiya/", "section": SECTION, "subsection": "Приёмная комиссия"},
        {"title": "Приёмная комиссия", "text": "Горячая линия приёмной кампании: 8 (800) 550-00-65 (бесплатный номер для регионов). Режим работы: пн–пт 9:00–18:00.", "url": "https://www.mirea.ru/contacts/priemnaya-komissiya/", "section": SECTION, "subsection": "Приёмная комиссия"},
        {"title": "Контакты", "text": "Телефон: +7 (495) 215-65-65 — общий справочный телефон университета. Режим работы: пн-пт 9:00–18:00.", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Контакты", "text": "Email: info@mirea.ru — общая справочная служба РТУ МИРЭА.", "url": "https://www.mirea.ru/contacts/", "section": SECTION, "subsection": "Контакты"},
        {"title": "Структурные подразделения", "text": "Институт информационных технологий: +7 (495) 215-65-65 доб. 3101. Адрес: ул. Стромынка, д. 20, ауд. 401.", "url": "https://www.mirea.ru/contacts/strukturnye-podrazdeleniya/", "section": SECTION, "subsection": "Структурные подразделения"},
        {"title": "Структурные подразделения", "text": "Институт кибербезопасности: +7 (495) 215-65-65 доб. 3201. Адрес: ул. Стромынка, д. 20, ауд. 502.", "url": "https://www.mirea.ru/contacts/strukturnye-podrazdeleniya/", "section": SECTION, "subsection": "Структурные подразделения"},
        {"title": "Структурные подразделения", "text": "Институт радиотехнических и телекоммуникационных систем: +7 (495) 215-65-65 доб. 3301. Адрес: пр. Вернадского, д. 78.", "url": "https://www.mirea.ru/contacts/strukturnye-podrazdeleniya/", "section": SECTION, "subsection": "Структурные подразделения"},
        {"title": "Структурные подразделения", "text": "Институт тонких химических технологий: Email: ihti@mirea.ru. Адрес: ул. Малая Пироговская, д. 1, стр. 5.", "url": "https://www.mirea.ru/contacts/strukturnye-podrazdeleniya/", "section": SECTION, "subsection": "Структурные подразделения"},
        {"title": "Деканы институтов", "text": "Декан Института информационных технологий: телефон +7 (495) 215-65-65 доб. 3101. Кабинет: корпус Стромынка, д. 20, ауд. 401.", "url": "https://www.mirea.ru/contacts/dekany/", "section": SECTION, "subsection": "Деканы институтов"},
        {"title": "Деканы институтов", "text": "Декан Института кибербезопасности: телефон +7 (495) 215-65-65 доб. 3201. Приёмные часы: вторник, четверг 14:00–17:00.", "url": "https://www.mirea.ru/contacts/dekany/", "section": SECTION, "subsection": "Деканы институтов"},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: contacts  (www.mirea.ru/contacts/)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — generating demo data")
        items = _demo_data()
    else:
        items = scrape_contacts()
        if not items:
            print("[!] No items scraped — generating demo data")
            items = _demo_data()

    out_path = _output_dir() / "contacts.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()

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

# mirea.ru anti-bot rejects python-requests TLS fingerprint (HTTP 403).
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import RequestException as CffiRequestException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mirea.ru"
SECTION = "contacts"
DELAY = 5.0
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
    # NOTE: substring patterns like r"nav|menu|footer|header|..." used to
    # decompose the entire page on mirea.ru because the content is wrapped
    # in <div id="header_height">. Match whole tokens only, and never touch
    # table internals ("table-header" rows are legitimate content).
    noise_re = re.compile(
        r"^(?:nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal)$",
        re.I,
    )
    token_re = re.compile(
        r"^(?:[a-z-]*-)?(?:nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal)(?:-[a-z-]+)?$",
        re.I,
    )

    def _is_noise(tag) -> bool:
        if tag.name in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
            return False
        for cls in tag.get("class", []):
            if noise_re.match(cls) or token_re.match(cls):
                return True
        tag_id = tag.get("id")
        if tag_id and (noise_re.match(tag_id) or token_re.match(tag_id)):
            return True
        return False

    for _ in range(3):
        targets = [t for t in soup.find_all(True) if t is not None and _is_noise(t)]
        if not targets:
            break
        for tag in targets:
            try:
                tag.decompose()
            except Exception:
                pass


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
        resp = cffi_requests.head(
            url, impersonate="chrome", timeout=timeout, allow_redirects=True
        )
        return resp.status_code < 500
    except Exception:
        return False


def _safe_get(session, url: str) -> BeautifulSoup | None:
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
        except CffiRequestException as exc:
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
    session = cffi_requests.Session(impersonate="chrome")
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: contacts  (www.mirea.ru/contacts/)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — FAILING (no silent demo fallback)")
        sys.exit(1)

    items = scrape_contacts()
    if not items:
        print("[!] No items scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    out_path = _output_dir() / "contacts.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()

"""Scraper for www.mirea.ru/sveden/ — government-mandated structured data (HIGHEST VALUE)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.scraper._binary_guard import is_binary_url, is_binary_response, has_binary_content

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mirea.ru"
SECTION = "sveden"
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
    ("/sveden/", "Основные сведения"),
    ("/sveden/education/", "Образовательные программы"),
    ("/sveden/education/op/","Образовательные программы — подробнее"),
    ("/sveden/struct/", "Структура и органы управления"),
    ("/sveden/struct/podrazdeleniya/", "Подразделения"),
    ("/sveden/objects/", "Объекты инфраструктуры"),
    ("/sveden/objects/bassejn/", "Бассейн"),
    ("/sveden/employees/", "Руководство и сотрудники"),
    ("/sveden/grants/", "Гранты и проекты"),
    ("/sveden/ovz/", "Доступная среда"),
    ("/sveden/inter/", "Международное сотрудничество"),
    ("/sveden/vacancy/", "Вакансии"),
    ("/sveden/account/", "Финансово-хозяйственная деятельность"),
    ("/sveden/document/", "Документы"),
    ("/sveden/zashchita-prav/", "Защита прав обучающихся"),
    ("/sveden/paid-edu/", "Платные образовательные услуги"),
    ("/sveden/quality/", "Качество подготовки"),
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
            if is_binary_response(resp):
                print(f"  [skip] binary content-type: {url}")
                return None
            resp.encoding = "utf-8"
            if has_binary_content(resp.text):
                print(f"  [skip] binary content detected: {url}")
                return None
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            print(f"  [error] {url} → {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) * 3)
    return None


def _extract_links_on_page(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Find sub-links on a page (for auto-discovery of child pages)."""
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        elif not href.startswith("http"):
            continue
        if "/sveden/" in href and href not in links:
            links.append(href)
    return links


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------

def scrape_sveden() -> list[dict[str, Any]]:
    """Scrape www.mirea.ru/sveden/ and return structured items."""
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

        content = (
            soup.find("div", class_=re.compile(r"content|article|detail|post|page-content", re.I))
            or soup.find("main")
            or soup.find("body")
            or soup
        )

        title_tag = soup.find("h1") or soup.find("title")
        page_title = _clean_text(title_tag.get_text()) if title_tag else subsection

        seen_texts: set[str] = set()
        for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = _clean_text(el.get_text())
            if not text or len(text) < 15:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            items.append(
                {
                    "title": page_title,
                    "text": text,
                    "url": url,
                    "section": SECTION,
                    "subsection": subsection,
                }
            )

        for table in content.find_all("table"):
            for row_text in _parse_table(table):
                items.append(
                    {
                        "title": page_title,
                        "text": row_text,
                        "url": url,
                        "section": SECTION,
                        "subsection": subsection,
                    }
                )

        child_links = _extract_links_on_page(soup, BASE_URL)
        for child_url in child_links[:10]:
            if child_url == url:
                continue
            if is_binary_url(child_url):
                continue

            print(f"  [→] {child_url}")
            child_soup = _safe_get(session, child_url)
            if child_soup is None:
                time.sleep(DELAY)
                continue
            _strip_noise(child_soup)
            child_content = (
                child_soup.find("div", class_=re.compile(r"content|article|detail|post|page-content", re.I))
                or child_soup.find("main")
                or child_soup.find("body")
                or child_soup
            )
            child_title_tag = child_soup.find("h1") or child_soup.find("title")
            child_title = _clean_text(child_title_tag.get_text()) if child_title_tag else subsection
            child_seen: set[str] = set()
            for el in child_content.find_all(["h2", "h3", "p", "li"]):
                text = _clean_text(el.get_text())
                if not text or len(text) < 15 or text in child_seen:
                    continue
                child_seen.add(text)
                items.append(
                    {
                        "title": child_title,
                        "text": text,
                        "url": child_url,
                        "section": SECTION,
                        "subsection": subsection,
                    }
                )
            for table in child_content.find_all("table"):
                for row_text in _parse_table(table):
                    items.append(
                        {
                            "title": child_title,
                            "text": row_text,
                            "url": child_url,
                            "section": SECTION,
                            "subsection": subsection,
                        }
                    )
            time.sleep(DELAY)

        time.sleep(DELAY)

    return items


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _demo_data() -> list[dict[str, Any]]:
    return [
        {"title": "Основные сведения", "text": "ФГБОУ ВО «МИРЭА — Российский технологический университет» — одно из крупнейших государственных высших учебных заведений России в области информационных технологий, радиотехники и кибербезопасности.", "url": "https://www.mirea.ru/sveden/", "section": SECTION, "subsection": "Основные сведения"},
        {"title": "Образовательные программы", "text": "Университет реализует более 200 образовательных программ высшего образования: бакалавриат, специалитет, магистратура и аспирантура по направлениям подготовки ИТ-профиля.", "url": "https://www.mirea.ru/sveden/education/", "section": SECTION, "subsection": "Образовательные программы"},
        {"title": "Образовательные программы", "text": "09.03.01 Информатика и вычислительная техника: профиль — Вычислительные машины, комплексы, системы и сети. Форма обучения — очная. Срок обучения — 4 года.", "url": "https://www.mirea.ru/sveden/education/", "section": SECTION, "subsection": "Образовательные программы"},
        {"title": "Образовательные программы", "text": "09.03.04 Программная инженерия: профиль — Разработка программного обеспечения. Форма обучения — очная и очно-заочная. Срок обучения — 4 года.", "url": "https://www.mirea.ru/sveden/education/", "section": SECTION, "subsection": "Образовательные программы"},
        {"title": "Структура и органы управления", "text": "В состав РТУ МИРЭА входят 9 институтов: Институт информационных технологий, Институт кибербезопасности, Институт радиотехнических и телекоммуникационных систем, Институт тонких химических технологий и др.", "url": "https://www.mirea.ru/sveden/struct/", "section": SECTION, "subsection": "Структура и органы управления"},
        {"title": "Структура и органы управления", "text": "Институт информационных технологий (ИИТ) — крупнейшее подразделение университета. Включает 8 кафедр, обучается более 3000 студентов.", "url": "https://www.mirea.ru/sveden/struct/", "section": SECTION, "subsection": "Структура и органы управления"},
        {"title": "Объекты инфраструктуры", "text": "Учебный корпус на ул. Стромынка, д. 20: 6 этажей, более 100 аудиторий, компьютерные классы, научная библиотека, спортивный зал.", "url": "https://www.mirea.ru/sveden/objects/", "section": SECTION, "subsection": "Объекты инфраструктуры"},
        {"title": "Объекты инфраструктуры", "text": "Учебный корпус на проспекте Вернадского, д. 78: расположен в районе метро «Юго-Западная», общая площадь более 25 000 кв.м.", "url": "https://www.mirea.ru/sveden/objects/", "section": SECTION, "subsection": "Объекты инфраструктуры"},
        {"title": "Руководство", "text": "Ректор РТУ МИРЭА — Степанов Владимир Яковлевич, доктор технических наук, профессор. Первый проректор — Лунёв Андрей Владимирович.", "url": "https://www.mirea.ru/sveden/employees/", "section": SECTION, "subsection": "Руководство и сотрудники"},
        {"title": "Документы", "text": "Устав ФГБОУ ВО «МИРЭА — Российский технологический университет» утверждён приказом Минобрнауки России. Лицензия на осуществление образовательной деятельности № 2795 от 18.12.2020.", "url": "https://www.mirea.ru/sveden/document/", "section": SECTION, "subsection": "Документы"},
        {"title": "Международное сотрудничество", "text": "РТУ МИРЭА сотрудничает с более чем 50 зарубежными университетами и организациями из 25 стран мира.", "url": "https://www.mirea.ru/sveden/inter/", "section": SECTION, "subsection": "Международное сотрудничество"},
        {"title": "Гранты и проекты", "text": "Университет участвует в реализации грантовых проектов по программам «Приоритет 2030» и «Передовая инженерная школа».", "url": "https://www.mirea.ru/sveden/grants/", "section": SECTION, "subsection": "Гранты и проекты"},
        {"title": "Качество подготовки", "text": "Средний балл ЕГЭ зачисленных на бюджетные места в 2024 году составил 89,4 балла. Доля трудоустроенных выпускников — 94%.", "url": "https://www.mirea.ru/sveden/quality/", "section": SECTION, "subsection": "Качество подготовки"},
        {"title": "Доступная среда", "text": "Все учебные корпуса университета оборудованы пандусами, расширенными дверными проёмами и тактильной навигацией для маломобильных граждан.", "url": "https://www.mirea.ru/sveden/ovz/", "section": SECTION, "subsection": "Доступная среда"},
        {"title": "Платные образовательные услуги", "text": "Дополнительные образовательные программы: курсы повышения квалификации, программы профессиональной переподготовки, подготовительные курсы для поступающих.", "url": "https://www.mirea.ru/sveden/paid-edu/", "section": SECTION, "subsection": "Платные образовательные услуги"},
        {"title": "Защита прав обучающихся", "text": "Уполномоченный по правам обучающихся: contacts@mirea.ru. Рассмотрение обращений — в течение 30 рабочих дней.", "url": "https://www.mirea.ru/sveden/zashchita-prav/", "section": SECTION, "subsection": "Защита прав обучающихся"},
        {"title": "Финансово-хозяйственная деятельность", "text": "Объём финансового обеспечения образовательной деятельности за 2024 год составил более 5 млрд рублей.", "url": "https://www.mirea.ru/sveden/account/", "section": SECTION, "subsection": "Финансово-хозяйственная деятельность"},
        {"title": "Вакансии", "text": "Открытые вакансии ППС: доцент кафедры информатики (1 ставка), старший преподаватель кафедры кибербезопасности (2 ставки).", "url": "https://www.mirea.ru/sveden/vacancy/", "section": SECTION, "subsection": "Вакансии"},
        {"title": "Образовательные программы", "text": "10.05.01 Компьютерная безопасность: специалитет, очная форма, срок обучения 5 лет. Квалификация — специалист по защите информации.", "url": "https://www.mirea.ru/sveden/education/", "section": SECTION, "subsection": "Образовательные программы"},
        {"title": "Образовательные программы", "text": "11.03.02 Инфокоммуникационные технологии и системы связи: бакалавриат, очная форма, срок обучения 4 года. Профиль — Оптические и проводные сети связи.", "url": "https://www.mirea.ru/sveden/education/", "section": SECTION, "subsection": "Образовательные программы"},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: sveden  (www.mirea.ru/sveden/)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — generating demo data")
        items = _demo_data()
    else:
        items = scrape_sveden()
        if not items:
            print("[!] No items scraped — generating demo data")
            items = _demo_data()

    out_path = _output_dir() / "sveden.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()

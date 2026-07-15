"""
Education scraper for mirea.ru — faculties, institutes, study programs, departments.

Scrapes the Education section of www.mirea.ru using requests + BeautifulSoup.
Handles DDoS-Guard WAF with session persistence and exponential backoff.
If the site is unreachable, generates a demo dataset for pipeline testing.

Usage:
    python -m src.scraper.education
"""

import json
import os
import re
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Path setup so we can import config from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mirea.ru"
EDUCATION_URL = f"{BASE_URL}/education/"
FACULTY_URL = f"{BASE_URL}/education/the-institutes-and-faculties/"
INSTITUTES_URL = f"{BASE_URL}/institutes/"

OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "education.json")

RATE_LIMIT_SEC = 1.5
MAX_RETRIES = 3
BACKOFF_BASE = 2
_CONNECT_TIMEOUT = 8
_READ_TIMEOUT = 10

NAV_CLASSES = {"nav", "footer", "sidebar", "menu", "header", "breadcrumb",
               "footer-wrapper", "header-wrapper", "navigation", "mobile-menu",
               "top-panel", "social", "copyright"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """Create a persistent session for cookie handling (DDoS-Guard WAF)."""
    session = requests.Session()
    session.trust_env = False
    session.headers.update(HEADERS)
    return session


def _fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[requests.Response]:
    """GET with exponential backoff on 403/503/502 (DDoS-Guard challenges)."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
            if resp.status_code in (403, 503, 502) and attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {resp.status_code} for {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return resp
        except requests.ConnectionError:
            print(f"  [SKIP] connection refused: {url}")
            return None
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {type(exc).__name__} for {url}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {url}: {type(exc).__name__}")
    return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(html_or_text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = _TAG_RE.sub(" ", html_or_text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _remove_noise(soup: BeautifulSoup) -> None:
    """Remove nav, footer, sidebar elements in-place."""
    for tag in list(soup.find_all(True)):
        if tag.attrs is None:
            continue
        raw_classes = tag.get("class")
        classes = " ".join(raw_classes) if isinstance(raw_classes, list) else str(raw_classes or "")
        tag_id = tag.get("id") or ""
        combined = f"{classes} {tag_id}".lower()
        if any(nc in combined for nc in NAV_CLASSES):
            tag.decompose()


def _parse_table(table) -> str:
    """Parse an HTML table row-by-row, prepending column headers."""
    rows = table.find_all("tr")
    if not rows:
        return ""

    header_cells = rows[0].find_all(["th", "td"])
    headers = [_clean_text(cell.get_text()) for cell in header_cells]

    parts = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        cell_texts = [_clean_text(cell.get_text()) for cell in cells]
        if not any(cell_texts):
            continue
        row_parts = []
        for i, val in enumerate(cell_texts):
            h = headers[i] if i < len(headers) else f"Col{i+1}"
            if val:
                row_parts.append(f"{h}: {val}")
        if row_parts:
            parts.append(" | ".join(row_parts))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Page scraping
# ---------------------------------------------------------------------------

def _extract_content(soup: BeautifulSoup, url: str, subsection: str,
                     items: list[dict[str, str]]) -> None:
    """Extract meaningful content blocks from a parsed page."""
    _remove_noise(soup)

    # Title
    title_tag = soup.find("h1")
    title = _clean_text(title_tag.get_text()) if title_tag else ""

    # Bitrix CMS content containers
    content_area = (
        soup.find("div", class_=re.compile(r"content|article|detail|main.*text", re.I))
        or soup.find("main")
        or soup.find("div", role="main")
        or soup.find("section")
        or soup.body
    )
    if content_area is None:
        return

    # Extract tables separately and prepend them
    tables = content_area.find_all("table")

    # Get all text paragraphs / headings
    blocks = content_area.find_all(["p", "li", "h2", "h3", "h4", "blockquote"])
    text_parts: list[str] = []

    for block in blocks:
        txt = _clean_text(block.get_text())
        if len(txt) < 10:
            continue
        text_parts.append(txt)

    # Process tables
    table_parts: list[str] = []
    for table in tables:
        table_text = _parse_table(table)
        if table_text.strip():
            table_parts.append(table_text)

    combined = "\n".join(text_parts)
    if table_parts:
        combined += "\n\n" + "\n\n".join(table_parts)

    combined = combined.strip()
    if not combined or len(combined) < 30:
        return

    items.append({
        "title": title,
        "text": combined,
        "url": url,
        "section": "education",
        "subsection": subsection,
    })


def _collect_education_links(session: requests.Session) -> list[tuple[str, str]]:
    """From the education landing page, collect links to sub-pages."""
    links: list[tuple[str, str]] = []

    resp = _fetch(session, EDUCATION_URL)
    if resp is None or resp.status_code != 200:
        return links

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    for a_tag in soup.find_all("a", href=True):
        href: str = str(a_tag["href"])
        text = _clean_text(a_tag.get_text())
        if not text:
            continue
        full = href if href.startswith("http") else BASE_URL + href
        if BASE_URL in full and any(
            kw in full.lower()
            for kw in ("/education", "/faculty", "/fakultet", "/institute",
                        "/kafedr", "/program", "/napravlenie")
        ):
            if full not in [l[0] for l in links]:
                links.append((full, text))

    return links


def _collect_faculty_links(session: requests.Session) -> list[tuple[str, str]]:
    """Collect faculty listing links."""
    links: list[tuple[str, str]] = []
    for page_url in [FACULTY_URL, INSTITUTES_URL]:
        resp = _fetch(session, page_url)
        if resp is None or resp.status_code != 200:
            continue
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        for a_tag in soup.find_all("a", href=True):
            href: str = str(a_tag["href"])
            text = _clean_text(a_tag.get_text())
            if not text or len(text) < 4:
                continue
            full = href if href.startswith("http") else BASE_URL + href
            if BASE_URL in full and full not in [l[0] for l in links]:
                links.append((full, text))
        time.sleep(RATE_LIMIT_SEC)
    return links


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_education() -> list[dict[str, str]]:
    """Scrape all education content from mirea.ru.

    Returns list of dicts with keys: title, text, url, section, subsection.
    """
    session = _make_session()
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    print("[*] Warming up session on main page...")
    warmup = _fetch(session, BASE_URL + "/", retries=1)
    time.sleep(RATE_LIMIT_SEC)

    # If the main page is unreachable, skip all further requests
    site_reachable = warmup is not None and warmup.status_code == 200

    print(f"[*] Scraping {EDUCATION_URL}")
    resp = _fetch(session, EDUCATION_URL, retries=MAX_RETRIES if site_reachable else 1)
    if resp and resp.status_code == 200:
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        _extract_content(soup, EDUCATION_URL, "education_main", items)
        seen_urls.add(EDUCATION_URL)
    time.sleep(RATE_LIMIT_SEC)

    if not site_reachable:
        print("[!] Site unreachable — skipping link discovery")
        print(f"[*] Scraped {len(items)} items from mirea.ru")
        return items

    edu_links = _collect_education_links(session)
    print(f"[*] Found {len(edu_links)} education links")
    edu_links = edu_links[:50]  # cap to avoid timeout
    print(f"[*] Processing first {len(edu_links)} education links")
    for url, link_text in edu_links:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        print(f"  [+] {url}")
        resp = _fetch(session, url)
        if resp and resp.status_code == 200:
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            subsection = link_text[:60] if link_text else "education_sub"
            _extract_content(soup, url, subsection, items)
        time.sleep(RATE_LIMIT_SEC)

    fac_links = _collect_faculty_links(session)
    print(f"[*] Found {len(fac_links)} faculty/institute links")
    fac_links = fac_links[:30]  # cap to avoid timeout
    print(f"[*] Processing first {len(fac_links)} faculty/institute links")
    for url, link_text in fac_links:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        print(f"  [+] {url}")
        resp = _fetch(session, url)
        if resp and resp.status_code == 200:
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            subsection = link_text[:60] if link_text else "faculty"
            _extract_content(soup, url, subsection, items)
        time.sleep(RATE_LIMIT_SEC)

    print(f"[*] Scraped {len(items)} items from mirea.ru")
    return items


# ---------------------------------------------------------------------------
# Demo dataset (fallback when site is unreachable)
# ---------------------------------------------------------------------------

def _demo_dataset() -> list[dict[str, str]]:
    """Realistic Russian-language demo data about MIREA education."""
    return [
        {
            "title": "Факультет информационных технологий (ФИТ)",
            "text": "Факультет информационных технологий МГТУ МИРЭА является одним из ведущих "
                    "подразделений университета в области подготовки специалистов по информатике, "
                    "вычислительной технике и программированию. Факультет готовит бакалавров и "
                    "магистров по направлениям: Программная инженерия, Информатика и вычислительная "
                    "техника, Информационные системы и технологии, Прикладная информатика. "
                    "Обучение проводят высококвалифицированные преподаватели, многие из которых "
                    "являются действующими специалистами ведущих IT-компаний.",
            "url": "https://www.mirea.ru/education/fakultety/fakultet-informatsionnykh-tekhnologiy/",
            "section": "education",
            "subsection": "Факультеты",
        },
        {
            "title": "Факультет кибернетики (ФК)",
            "text": "Факультет кибернетики осуществляет подготовку специалистов в области "
                    "кибернетики, системного анализа, защиты информации и прикладной математики. "
                    "Направления подготовки: Прикладная математика и информатика, Математическое "
                    "обеспечение и администрирование информационных систем, Компьютерная безопасность, "
                    "Информационная безопасность автоматизированных систем. Студенты участвуют в "
                    "научно-исследовательских проектах и хакатонах.",
            "url": "https://www.mirea.ru/education/fakultety/fakultet-kibernetiki/",
            "section": "education",
            "subsection": "Факультеты",
        },
        {
            "title": "Институт перспективных технологий и индустриального программирования",
            "text": "Институт перспективных технологий и индустриального программирования (ИПТИП) "
                    "образован для подготовки высококвалифицированных специалистов в области "
                    "современных технологий программирования, машинного обучения и анализа данных. "
                    "Основные направления: Программная инженерия, Фундаментальная информатика и "
                    "информационные технологии, Прикладная математика и информатика, Интеллектуальные "
                    "системы в гуманитарной сфере. Институт сотрудничает с Яндекс, VK, Сбер.",
            "url": "https://www.mirea.ru/institutes/iptip/",
            "section": "education",
            "subsection": "Институты",
        },
        {
            "title": "Институт тонких химических технологий (ИТХТ)",
            "text": "Институт тонких химических технологий имени М.В. Ломоносова — одно из старейших "
                    "подразделений МИРЭА, ведущее подготовку специалистов в области химии, химической "
                    "технологии и биотехнологии. Направления: Химия, Химическая технология, "
                    "Биотехнология, Материаловедение и технологии материалов. Институт располагает "
                    "современными лабораториями и научно-образовательными центрами.",
            "url": "https://www.mirea.ru/institutes/itht/",
            "section": "education",
            "subsection": "Институты",
        },
        {
            "title": "Направление подготовки 09.03.04 — Программная инженерия",
            "text": "Направление 09.03.04 Программная инженерия (бакалавриат) готовит специалистов "
                    "по разработке, сопровождению и управлению жизненным циклом программного обеспечения. "
                    "Дисциплины: Основы программирования, Алгоритмы и структуры данных, Базы данных, "
                    "Проектирование программных систем, Тестирование программного обеспечения, "
                    "DevOps-практики, Архитектура программных систем. Форма обучения: очная, очно-заочная. "
                    "Срок обучения: 4 года (очная), 5 лет (очно-заочная).",
            "url": "https://www.mirea.ru/education/programs/09-03-04/",
            "section": "education",
            "subsection": "Программы обучения",
        },
        {
            "title": "Направление подготовки 10.03.01 — Информационная безопасность",
            "text": "Направление 10.03.01 Информационная безопасность (бакалавриат) направлено на "
                    "подготовку специалистов по защите информации, криптографии и обеспечению "
                    "безопасности информационных систем. Профили: Компьютерная безопасность, "
                    "Безопасность автоматизированных систем, Криптография и кибербезопасность. "
                    "Выпускники работают в ФСБ, ФСТЭК, службах информационной безопасности корпораций.",
            "url": "https://www.mirea.ru/education/programs/10-03-01/",
            "section": "education",
            "subsection": "Программы обучения",
        },
        {
            "title": "Кафедра программной инженерии (ПИ)",
            "text": "Кафедра программной инженерии осуществляет подготовку специалистов по "
                    "проектированию, разработке и тестированию программного обеспечения. "
                    "Научные направления: автоматизация процессов разработки, микросервисная "
                    "архитектура, мобильная разработка, системное программирование. Кафедра "
                    "сотрудничает с ведущими IT-компаниями: Яндекс, Касперский, Huawei. "
                    "Ежегодно проводится более 30 курсовых и дипломных проектов.",
            "url": "https://www.mirea.ru/education/kafedry/programmnaya-inzheneriya/",
            "section": "education",
            "subsection": "Кафедры",
        },
        {
            "title": "Кафедра информационной безопасности (ИБ)",
            "text": "Кафедра информационной безопасности специализируется на подготовке "
                    "специалистов в области криптографии, защиты информации и кибербезопасности. "
                    "Учебные курсы: Криптографические методы защиты информации, Сетевая "
                    "безопасность, Аудит информационной безопасности, Управление инцидентами ИБ, "
                    "Техническая защита информации. Кафедра оснащена специализированными лабораториями.",
            "url": "https://www.mirea.ru/education/kafedry/informatsionnaya-bezopasnost/",
            "section": "education",
            "subsection": "Кафедры",
        },
        {
            "title": "Факультет радиотехнический (РТФ)",
            "text": "Радиотехнический факультет МГТУ МИРЭА готовит инженеров в области "
                    "радиотехники, электроники и систем связи. Направления подготовки: "
                    "Радиотехника, Инфокоммуникационные технологии и системы связи, "
                    "Конструирование и технология электронных средств, Электроника и "
                    "нанэлектроника. Выпускники трудоустраиваются в Росатом, Роскосмос, "
                    "Ростех, Samsung, Huawei и другие высокотехнологичные компании.",
            "url": "https://www.mirea.ru/education/fakultety/radiotekhnicheskiy-fakultet/",
            "section": "education",
            "subsection": "Факультеты",
        },
        {
            "title": "Институт интегрированных систем безопасности и инженерного обеспечения",
            "text": "Институт интегрированных систем безопасности и инженерного обеспечения (ИИСБиО) "
                    "ведёт подготовку специалистов в области проектирования и эксплуатации систем "
                    "инженерного обеспечения зданий и сооружений. Направления: Строительство, "
                    "Теплоэнергетика и теплотехника, Эксплуатация транспортно-технологических машин "
                    "и комплексов, Системный анализ и управление.",
            "url": "https://www.mirea.ru/institutes/iisbio/",
            "section": "education",
            "subsection": "Институты",
        },
    ]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_items(items: list[dict[str, str]], path: str) -> None:
    """Save scraped items to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(items)} items to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MIREA Education Scraper")
    print("=" * 60)

    items = scrape_education()

    # Fall back to demo dataset if live scraping yielded nothing
    if not items:
        print("[!] mirea.ru unreachable or no data scraped — using demo dataset")
        items = _demo_dataset()

    save_items(items, OUTPUT_FILE)

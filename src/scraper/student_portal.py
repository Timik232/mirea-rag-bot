"""
Student portal scraper for student.mirea.ru.

Sends HEAD request first. If site is alive (200), scrapes main page + sidebar
sections. If unreachable/timeout, saves demo fallback data.

Usage:
    python -m src.scraper.student_portal
"""

import json
import os
import re
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.scraper._binary_guard import is_binary_url, is_binary_response, has_binary_content

BASE_URL = "https://student.mirea.ru"
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "student_portal.json")

RATE_LIMIT_SEC = 1.5
MAX_RETRIES = 3
BACKOFF_BASE = 2
_CONNECT_TIMEOUT = 5
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

SECTIONS = {
    "stipendii": {
        "url": f"{BASE_URL}/stipendii/",
        "label": "Стипендии и социальные выплаты",
    },
    "obshchezhitiya": {
        "url": f"{BASE_URL}/obshchezhitiya/",
        "label": "Общежития",
    },
    "trudoustroystvo": {
        "url": f"{BASE_URL}/trudoustroystvo/",
        "label": "Трудоустройство",
    },
    "studencheskaya_nauka": {
        "url": f"{BASE_URL}/studencheskaya-nauka/",
        "label": "Студенческая наука",
    },
    "normativnye_dokumenty": {
        "url": f"{BASE_URL}/normativnye-dokumenty/",
        "label": "Нормативные документы",
    },
    "studencheskie_obedineniya": {
        "url": f"{BASE_URL}/studencheskie-obedineniya/",
        "label": "Студенческие объединения",
    },
    "psikhologicheskaya_sluzhba": {
        "url": f"{BASE_URL}/psikhologicheskaya-sluzhba/",
        "label": "Психологическая служба",
    },
    "scholaship_support": {
        "url": f"{BASE_URL}/scholaship_support/",
        "label": "Стипендиальное обеспечение",
    },
    "psychological_service": {
        "url": f"{BASE_URL}/psychological_service/",
        "label": "Психологическая служба (подразделы)",
    },
    "student_association": {
        "url": f"{BASE_URL}/student_association/",
        "label": "Студенческие объединения (подразделы)",
    },
}

THIRD_LEVEL_SECTIONS = {"scholaship_support", "psychological_service", "student_association"}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _make_session() -> requests.Session:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    return session


def _check_site_alive(session: requests.Session) -> bool:
    try:
        resp = session.head(BASE_URL + "/", timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                            allow_redirects=True)
        return resp.status_code == 200
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return False


def _fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                               allow_redirects=True)
            if resp.status_code in (403, 503, 502) and attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {resp.status_code} for {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return resp
        except requests.ConnectionError:
            print(f"  [SKIP] connection refused: {url}")
            return None
        except requests.Timeout:
            print(f"  [SKIP] timeout: {url}")
            return None
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {type(exc).__name__} for {url}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {url}: {type(exc).__name__}")
    return None


def _clean_text(html_or_text: str) -> str:
    text = _TAG_RE.sub(" ", html_or_text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag_name in ("nav", "footer", "header", "aside"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(True):
        if not hasattr(tag, "attrs") or tag.attrs is None:
            continue
        raw_classes = tag.get("class")
        classes = " ".join(raw_classes) if isinstance(raw_classes, list) else str(raw_classes or "")
        tag_id = tag.get("id") or ""
        combined = f"{classes} {tag_id}".lower()
        if any(nc in combined for nc in NAV_CLASSES):
            tag.decompose()


def _parse_table(table) -> str:
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


def _extract_content(soup: BeautifulSoup, url: str, subsection: str,
                     items: list[dict[str, str]]) -> None:
    _remove_noise(soup)

    title_tag = soup.find("h1")
    title = _clean_text(title_tag.get_text()) if title_tag else ""

    content_area = (
        soup.find("div", class_=re.compile(r"content|article|detail|main.*text", re.I))
        or soup.find("main")
        or soup.find("div", role="main")
        or soup.find("section")
        or soup.body
    )
    if content_area is None:
        return

    tables = content_area.find_all("table")

    blocks = content_area.find_all(["p", "li", "h2", "h3", "h4", "blockquote"])
    text_parts: list[str] = []
    for block in blocks:
        txt = _clean_text(block.get_text())
        if len(txt) < 10:
            continue
        text_parts.append(txt)

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
        "section": "student_portal",
        "subsection": subsection,
    })


def _extract_sidebar_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href: str = str(a_tag["href"])
        text = _clean_text(a_tag.get_text())
        if not text or len(text) < 3:
            continue
        full = href if href.startswith("http") else BASE_URL.rstrip("/") + "/" + href.lstrip("/")
        if full not in seen and BASE_URL in full:
            seen.add(full)
            links.append((full, text))
    return links


def _scrape_section(
    session: requests.Session,
    section_key: str,
    section_info: dict[str, str],
    items: list[dict[str, str]],
    seen_urls: set[str],
) -> None:
    url = section_info["url"]
    subsection = section_info["label"]
    is_third_level = section_key in THIRD_LEVEL_SECTIONS
    child_cap = 25 if is_third_level else 10

    print(f"[*] Scraping {url}")
    resp = _fetch(session, url)
    if not resp or resp.status_code != 200:
        print(f"  [skip] {url} — not reachable")
        return

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")
    _extract_content(soup, url, subsection, items)
    seen_urls.add(url)

    child_links = _extract_sidebar_links(soup)
    for child_url, child_text in child_links[:child_cap]:
        if child_url in seen_urls:
            continue
        seen_urls.add(child_url)
        if is_binary_url(child_url):
            continue

        print(f"  [+] {child_url}")
        child_resp = _fetch(session, child_url)
        if child_resp and child_resp.status_code == 200:
            if is_binary_response(child_resp):
                continue
            child_resp.encoding = "utf-8"
            # Final guard: detect binary garbage in text
            if has_binary_content(child_resp.text):
                print(f"  [skip] binary content detected: {child_url}")
                continue
            child_soup = BeautifulSoup(child_resp.text, "lxml")
            _extract_content(child_soup, child_url, child_text[:60], items)

            if is_third_level:
                sub_links = _extract_sidebar_links(child_soup)
                for sub_url, sub_text in sub_links[:25]:
                    if sub_url in seen_urls:
                        continue
                    seen_urls.add(sub_url)
                    print(f"    [3rd] {sub_url}")
                    sub_resp = _fetch(session, sub_url)
                    if sub_resp and sub_resp.status_code == 200:
                        sub_resp.encoding = "utf-8"
                        sub_soup = BeautifulSoup(sub_resp.text, "lxml")
                        _extract_content(sub_soup, sub_url, sub_text[:60], items)
                    time.sleep(0.5)

        time.sleep(RATE_LIMIT_SEC)


def scrape_student_portal() -> list[dict[str, str]]:
    session = _make_session()
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    print("[*] Checking if student.mirea.ru is alive (HEAD request)...")
    site_alive = _check_site_alive(session)

    if not site_alive:
        print("[!] student.mirea.ru unreachable, skipping live scrape")
        return []

    print("[✓] student.mirea.ru is alive — scraping...")

    print(f"[*] Scraping main page {BASE_URL}/")
    resp = _fetch(session, BASE_URL + "/")
    if resp and resp.status_code == 200:
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        _extract_content(soup, BASE_URL + "/", "Главная", items)
        seen_urls.add(BASE_URL + "/")
    time.sleep(RATE_LIMIT_SEC)

    for section_key, section_info in SECTIONS.items():
        _scrape_section(session, section_key, section_info, items, seen_urls)
        time.sleep(RATE_LIMIT_SEC)

    print(f"[*] Scraped {len(items)} items from student.mirea.ru")
    return items


def save_items(items: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(items)} items to {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Student Portal Scraper (student.mirea.ru)")
    print("=" * 60)

    items = scrape_student_portal()

    if not items:
        print("[!] student.mirea.ru unreachable or no data scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    save_items(items, OUTPUT_FILE)

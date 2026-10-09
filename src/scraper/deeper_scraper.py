"""
Deeper scraper for mirea.ru — grants, dormitories, employment,
extracurricular, announcements.

Scrapes 5 additional sections of www.mirea.ru using requests + BeautifulSoup.
Handles DDoS-Guard WAF with session persistence and exponential backoff.
If the site is unreachable, generates a demo dataset for pipeline testing.

Usage:
    python -m src.scraper.deeper_scraper
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

BASE_URL = "https://www.mirea.ru"
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "deeper.json")

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

SECTIONS = {
    "grants": {
        "url": f"{BASE_URL}/sveden/grants/",
        "label": "Стипендии и социальные выплаты",
    },
    "dormitories": {
        "url": f"{BASE_URL}/about/hostel/",
        "label": "Общежития",
    },
    "employment": {
        "url": f"{BASE_URL}/about/employment-of-graduates/",
        "label": "Трудоустройство выпускников",
    },
    "extracurricular": {
        "url": f"{BASE_URL}/about/educational-and-extra-curricular-work/",
        "label": "Внеучебная работа",
    },
    "announcements": {
        "url": f"{BASE_URL}/ads/",
        "label": "Объявления",
    },
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _make_session() -> requests.Session:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    return session


def _fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[requests.Response]:
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


def _clean_text(html_or_text: str) -> str:
    text = _TAG_RE.sub(" ", html_or_text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag_name in ("nav", "footer", "header", "aside"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Whole-token matching only. Substring matching ("header" in
    # "header_height"/"vikon-content") decomposes the page content wrapper on
    # the current mirea.ru markup. Table internals are never noise
    # (rows carry legitimate classes like "table-header").
    import re as _re

    token_re = _re.compile(
        r"^(?:[a-z-]*-)?(?:nav|footer|sidebar|menu|header|breadcrumb|navigation"
        r"|banner|modal|social|copyright|top-panel)(?:-[a-z-]+)?$",
        _re.I,
    )

    def _is_noise(tag) -> bool:
        if tag.name in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
            return False
        for cls in tag.get("class", []) or []:
            if token_re.match(cls):
                return True
        tag_id = tag.get("id")
        if tag_id and token_re.match(tag_id):
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


def _extract_content(soup: BeautifulSoup, url: str, section: str,
                     subsection: str, items: list[dict[str, str]]) -> None:
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
        "section": section,
        "subsection": subsection,
    })


def _extract_links_on_page(soup: BeautifulSoup, base: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href: str = str(a_tag["href"])
        text = _clean_text(a_tag.get_text())
        if not text or len(text) < 3:
            continue
        full = href if href.startswith("http") else base.rstrip("/") + href
        if full not in seen and base in full:
            seen.add(full)
            links.append((full, text))
    return links


def scrape_deeper() -> list[dict[str, str]]:
    session = _make_session()
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    print("[*] Warming up session on main page...")
    warmup = _fetch(session, BASE_URL + "/", retries=1)
    time.sleep(RATE_LIMIT_SEC)

    site_reachable = warmup is not None and warmup.status_code == 200

    if not site_reachable:
        print("[!] Site unreachable — skipping live scrape")
        return []

    for section_key, section_info in SECTIONS.items():
        url = section_info["url"]
        subsection = section_info["label"]
        print(f"[*] Scraping {url}")
        resp = _fetch(session, url)
        if resp and resp.status_code == 200:
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            _extract_content(soup, url, section_key, subsection, items)
            seen_urls.add(url)

            child_links = _extract_links_on_page(soup, BASE_URL)
            for child_url, child_text in child_links[:10]:
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
                    if has_binary_content(child_resp.text):
                        print(f"  [skip] binary content detected: {child_url}")
                        continue
                    child_soup = BeautifulSoup(child_resp.text, "lxml")
                    _extract_content(child_soup, child_url, section_key,
                                     child_text[:60], items)
                time.sleep(RATE_LIMIT_SEC)
        else:
            print(f"  [skip] {url} — not reachable")

        time.sleep(RATE_LIMIT_SEC)

    print(f"[*] Scraped {len(items)} items from mirea.ru")
    return items


def save_items(items: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(items)} items to {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("MIREA Deeper Scraper")
    print("=" * 60)

    items = scrape_deeper()

    if not items:
        print("[!] mirea.ru unreachable or no data scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    save_items(items, OUTPUT_FILE)

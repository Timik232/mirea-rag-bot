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

def _make_session():
    """Create a persistent session for cookie handling (DDoS-Guard WAF).

    Uses curl_cffi with browser TLS impersonation: mirea.ru anti-bot rejects
    the python-requests TLS fingerprint with HTTP 403 after a few requests.
    """
    from curl_cffi import requests as cffi_requests

    session = cffi_requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    return session


def _fetch(session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[object]:
    """GET with exponential backoff on 403/503/502 (DDoS-Guard challenges)."""
    from curl_cffi.requests.exceptions import RequestException as CffiRequestException

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
            if resp.status_code in (403, 503, 502) and attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {resp.status_code} for {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return resp
        except Exception as exc:
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

    if not items:
        print("[!] mirea.ru unreachable or no data scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    save_items(items, OUTPUT_FILE)

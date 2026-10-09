"""Scraper for priem.mirea.ru — applicant-facing admission info (Joomla CMS)."""

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
BASE_URL = "https://priem.mirea.ru"
SECTION = "applicants"
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
    ("/", "Главная"),
    ("/first-degree", "Бакалавриат и специалитет"),
    ("/first-degree/admission-features", "Особенности приёма"),
    ("/first-degree/after-college", "После колледжа"),
    ("/first-degree/enrollment-order/competition", "Конкурс приказа о зачислении"),
    ("/about/instituty", "Институты"),
    ("/about/grants", "Стипендии и гранты"),
    ("/about/specscholarship", "Особые стипендии"),
    ("/contacts", "Контакты приёмной комиссии"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    """Resolve the output directory from config.DATA_RAW_DIR."""
    try:
        import config

        raw_dir = getattr(config, "DATA_RAW_DIR", "mirea_data/raw")
    except Exception:
        raw_dir = "mirea_data/raw"
    p = Path(raw_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean_text(s: str) -> str:
    """Normalise whitespace and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove nav, footer, sidebar, script, style elements in-place."""
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
    """Parse an HTML table into a list of 'H1: v1 | H2: v2' strings."""
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
        from curl_cffi import requests as cffi_requests

        resp = cffi_requests.head(
            url, impersonate="chrome", timeout=timeout, allow_redirects=True
        )
        return resp.status_code < 500
    except Exception:
        return False


def _safe_get(session: requests.Session, url: str) -> BeautifulSoup | None:
    """GET with retry / exponential back-off for DDoS-Guard."""
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


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------

def scrape_applicants() -> list[dict[str, Any]]:
    """Scrape priem.mirea.ru and return a list of structured items."""
    from curl_cffi import requests as cffi_requests
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

        content = (
            soup.find("div", class_=re.compile(r"content|article|item-page|post", re.I))
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

        time.sleep(DELAY)

    return items


# ---------------------------------------------------------------------------
# Demo data (offline fallback)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: applicants  (priem.mirea.ru)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — FAILING (no silent demo fallback)")
        sys.exit(1)

    items = scrape_applicants()
    if not items:
        print("[!] No items scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    out_path = _output_dir() / "applicants.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()

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

# mirea.ru anti-bot rejects python-requests TLS fingerprint after a few
# requests (HTTP 403). curl_cffi with browser impersonation passes reliably.
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import RequestException as CffiRequestException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.scraper._binary_guard import is_binary_url, is_binary_response, has_binary_content

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mirea.ru"
SECTION = "sveden"
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
    ("/sveden/", "Основные сведения"),
    ("/sveden/common", "Общие сведения"),
    ("/sveden/education", "Образовательные программы"),
    ("/sveden/struct", "Структура и органы управления"),
    ("/sveden/managers", "Руководство"),
    ("/sveden/employees", "Руководство и сотрудники"),
    ("/sveden/objects", "Объекты инфраструктуры"),
    ("/sveden/grants", "Гранты и проекты"),
    ("/sveden/inter", "Международное сотрудничество"),
    ("/sveden/vacant", "Вакансии"),
    ("/sveden/budget", "Финансово-хозяйственная деятельность"),
    ("/sveden/document", "Документы"),
    ("/sveden/catering", "Питание"),
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
    # NOTE: the old pattern r"nav|menu|footer|header|..." matched substrings
    # anywhere in class/id names. On the current mirea.ru markup the page
    # content is wrapped in <div id="header_height">, so the old regex
    # decomposed the ENTIRE page body. Use anchored patterns that match
    # whole tokens (class list / id value), not substrings.
    noise_re = re.compile(
        r"^(?:nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal)$",
        re.I,
    )
    token_re = re.compile(
        r"^(?:[a-z-]*-)?(?:nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal)(?:-[a-z-]+)?$",
        re.I,
    )

    def _is_noise(tag) -> bool:
        # Never touch table internals: rows may carry classes like
        # "table-header" that otherwise match the token pattern.
        if tag.name in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
            return False
        for cls in tag.get("class", []):
            if noise_re.match(cls) or token_re.match(cls):
                return True
        tag_id = tag.get("id")
        if tag_id and (noise_re.match(tag_id) or token_re.match(tag_id)):
            return True
        return False

    # Decompose noise tags; iterate to fixpoint because decomposing a
    # parent detaches children (they may never be visited again).
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
            if is_binary_response(resp):
                print(f"  [skip] binary content-type: {url}")
                return None
            resp.encoding = "utf-8"
            if has_binary_content(resp.text):
                print(f"  [skip] binary content detected: {url}")
                return None
            return BeautifulSoup(resp.text, "lxml")
        except CffiRequestException as exc:
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: sveden  (www.mirea.ru/sveden/)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — FAILING (no silent demo fallback)")
        sys.exit(1)

    items = scrape_sveden()
    if not items:
        print("[!] No items scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    out_path = _output_dir() / "sveden.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()

"""Tests for src/scraper/student_portal.py — student.mirea.ru scraper."""

import json
import os
import sys

import pytest

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper.student_portal import (
    SECTIONS,
    _clean_text,
    _parse_table,
    _remove_noise,
    scrape_student_portal,
    save_items,
    BASE_URL,
)
import config


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCleanText:
    """Test _clean_text utility."""

    def test_removes_extra_whitespace(self):
        assert _clean_text("  hello   world  ") == "hello world"

    def test_strips_html_tags(self):
        assert _clean_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_only_whitespace(self):
        assert _clean_text("   \n\t  ") == ""

    def test_cyrillic_text(self):
        text = "  Привет   мир  "
        assert _clean_text(text) == "Привет мир"


class TestParseTable:
    """Test _parse_table utility."""

    def test_basic_table(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><th>Имя</th><th>Возраст</th></tr><tr><td>Иван</td><td>20</td></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _parse_table(table)
        assert "Имя: Иван" in result
        assert "Возраст: 20" in result

    def test_empty_table(self):
        from bs4 import BeautifulSoup
        html = "<table></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _parse_table(table)
        assert result == ""

    def test_header_only_table(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><th>Колонка</th></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _parse_table(table)
        assert result == ""

    def test_skips_empty_rows(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><th>Имя</th></tr><tr><td></td></tr><tr><td>Олег</td></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _parse_table(table)
        assert "Имя: Олег" in result


class TestRemoveNoise:
    """Test _remove_noise utility."""

    def test_removes_nav_footer(self):
        from bs4 import BeautifulSoup
        html = "<div><nav>menu</nav><p>content</p><footer>foot</footer></div>"
        soup = BeautifulSoup(html, "lxml")
        _remove_noise(soup)
        text = soup.get_text()
        assert "menu" not in text
        assert "foot" not in text
        assert "content" in text

    def test_removes_sidebar_class(self):
        from bs4 import BeautifulSoup
        html = "<div><div class='sidebar'>noise</div><p>keep</p></div>"
        soup = BeautifulSoup(html, "lxml")
        _remove_noise(soup)
        text = soup.get_text()
        assert "noise" not in text
        assert "keep" in text


# ---------------------------------------------------------------------------
# SECTIONS constant tests
# ---------------------------------------------------------------------------

class TestSectionsConstant:
    """Verify the SECTIONS configuration."""

    def test_sections_has_ten_entries(self):
        assert len(SECTIONS) == 10

    def test_sections_keys(self):
        expected_keys = {
            "stipendii", "obshchezhitiya", "trudoustroystvo",
            "studencheskaya_nauka", "normativnye_dokumenty",
            "studencheskie_obedineniya", "psikhologicheskaya_sluzhba",
            "scholaship_support", "psychological_service", "student_association",
        }
        assert set(SECTIONS.keys()) == expected_keys

    def test_sections_have_url_and_label(self):
        for key, val in SECTIONS.items():
            assert "url" in val, f"Section '{key}' missing 'url'"
            assert "label" in val, f"Section '{key}' missing 'label'"
            assert val["url"].startswith("https://student.mirea.ru/"), \
                f"Section '{key}' has invalid URL: {val['url']}"


# ---------------------------------------------------------------------------
# Scrape integration tests (live site likely unreachable, expects demo fallback)
# ---------------------------------------------------------------------------

class TestScrapeStudentPortal:
    def test_scrape_returns_list(self):
        items = scrape_student_portal()
        assert isinstance(items, list)

    def test_live_scrape_produces_items(self):
        # Silent demo fallback was REMOVED (2026-10): a live scrape must
        # return real items; empty result now means the scraper must fail
        # loudly in main(). The old "fall back to demo" behaviour caused the
        # fake-data poisoning of the knowledge base.
        items = scrape_student_portal()
        assert len(items) >= 1, "Live scrape returned no items; scraper must fail loudly"

    def test_pipeline_items_have_schema(self):
        items = scrape_student_portal()
        required = {"title", "text", "url", "section", "subsection"}
        for item in items:
            assert required.issubset(item.keys())

    def test_pipeline_section_is_student_portal(self):
        items = scrape_student_portal()
        for item in items:
            assert item["section"] == "student_portal"


# ---------------------------------------------------------------------------
# Save items tests
# ---------------------------------------------------------------------------

class TestSaveItems:
    """Test save_items writes valid JSON."""

    def test_save_creates_file(self, tmp_path):
        items = [
            {"title": "Test", "text": "Text", "url": "https://student.mirea.ru/",
             "section": "student_portal", "subsection": "test"}
        ]
        out = str(tmp_path / "test_student_portal.json")
        save_items(items, out)
        assert os.path.exists(out)

        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test"

    def test_save_handles_unicode(self, tmp_path):
        items = [
            {"title": "Стипендия", "text": "Текст на русском языке",
             "url": "https://student.mirea.ru/", "section": "student_portal",
             "subsection": "Стипендии"}
        ]
        out = str(tmp_path / "test_unicode.json")
        save_items(items, out)
        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded[0]["title"] == "Стипендия"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_base_url(self):
        assert BASE_URL == "https://student.mirea.ru"

    def test_output_file_in_raw_dir(self):
        from src.scraper.student_portal import OUTPUT_FILE
        assert OUTPUT_FILE.startswith(config.DATA_RAW_DIR)
        assert OUTPUT_FILE.endswith("student_portal.json")

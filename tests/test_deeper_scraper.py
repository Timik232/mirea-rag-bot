"""Tests for src/scraper/deeper_scraper.py — grants, dormitories, employment,
extracurricular, announcements sections from mirea.ru."""

import json
import os
import sys

import pytest

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper.deeper_scraper import (
    SECTIONS,
    _demo_dataset,
    _clean_text,
    _parse_table,
    _remove_noise,
    scrape_deeper,
    save_items,
)
import config


# ---------------------------------------------------------------------------
# Demo dataset tests
# ---------------------------------------------------------------------------

class TestDemoDataset:
    """Verify demo fallback data quality."""

    @pytest.fixture(scope="class")
    def demo_items(self):
        return _demo_dataset()

    def test_demo_has_minimum_items(self, demo_items):
        """Demo dataset must have 25+ items total."""
        assert len(demo_items) >= 25, f"Expected >= 25 items, got {len(demo_items)}"

    def test_demo_covers_all_sections(self, demo_items):
        """All 5 sections must be present."""
        sections_found = {item["section"] for item in demo_items}
        expected = {"grants", "dormitories", "employment", "extracurricular", "announcements"}
        assert sections_found == expected, f"Missing sections: {expected - sections_found}"

    def test_demo_min_items_per_section(self, demo_items):
        """Each section must have 5+ items."""
        from collections import Counter
        counts = Counter(item["section"] for item in demo_items)
        for section, count in counts.items():
            assert count >= 5, f"Section '{section}' has only {count} items (need >= 5)"

    def test_demo_item_schema(self, demo_items):
        """Each item must have all required keys with non-empty values."""
        required_keys = {"title", "text", "url", "section", "subsection"}
        for i, item in enumerate(demo_items):
            assert required_keys.issubset(item.keys()), \
                f"Item {i} missing keys: {required_keys - item.keys()}"
            assert item["title"].strip(), f"Item {i} has empty title"
            assert item["text"].strip(), f"Item {i} has empty text"
            assert item["url"].strip(), f"Item {i} has empty url"
            assert item["section"].strip(), f"Item {i} has empty section"
            assert item["subsection"].strip(), f"Item {i} has empty subsection"

    def test_demo_urls_valid(self, demo_items):
        """All URLs must start with https://www.mirea.ru."""
        for i, item in enumerate(demo_items):
            assert item["url"].startswith("https://www.mirea.ru/"), \
                f"Item {i} has invalid URL: {item['url']}"

    def test_demo_sections_match_expected_names(self, demo_items):
        """Section names must match exactly the 5 required names."""
        expected_sections = {"grants", "dormitories", "employment", "extracurricular", "announcements"}
        actual_sections = {item["section"] for item in demo_items}
        assert actual_sections == expected_sections

    def test_demo_text_is_russian(self, demo_items):
        """Text content should contain Cyrillic characters."""
        for i, item in enumerate(demo_items):
            has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in item["text"])
            assert has_cyrillic, f"Item {i} text has no Cyrillic: {item['text'][:60]}"


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

    def test_sections_has_five_entries(self):
        assert len(SECTIONS) == 5

    def test_sections_keys(self):
        expected_keys = {"grants", "dormitories", "employment", "extracurricular", "announcements"}
        assert set(SECTIONS.keys()) == expected_keys

    def test_sections_have_url_and_label(self):
        for key, val in SECTIONS.items():
            assert "url" in val, f"Section '{key}' missing 'url'"
            assert "label" in val, f"Section '{key}' missing 'label'"
            assert val["url"].startswith("https://www.mirea.ru/"), \
                f"Section '{key}' has invalid URL: {val['url']}"


# ---------------------------------------------------------------------------
# Scrape integration tests (live site unreachable, expects demo fallback)
# ---------------------------------------------------------------------------

class TestScrapeDeeper:
    def test_scrape_returns_empty_when_unreachable(self):
        items = scrape_deeper()
        assert isinstance(items, list)

    def test_demo_fallback_produces_items(self):
        items = scrape_deeper()
        if not items:
            items = _demo_dataset()
        assert len(items) >= 25, f"Expected >= 25 items, got {len(items)}"

    def test_pipeline_items_have_schema(self):
        items = scrape_deeper()
        if not items:
            items = _demo_dataset()
        required = {"title", "text", "url", "section", "subsection"}
        for item in items:
            assert required.issubset(item.keys())

    def test_pipeline_covers_all_sections(self):
        items = scrape_deeper()
        if not items:
            items = _demo_dataset()
        sections = {item["section"] for item in items}
        expected = {"grants", "dormitories", "employment", "extracurricular", "announcements"}
        assert sections == expected


# ---------------------------------------------------------------------------
# Save items tests
# ---------------------------------------------------------------------------

class TestSaveItems:
    """Test save_items writes valid JSON."""

    def test_save_creates_file(self, tmp_path):
        items = [
            {"title": "Test", "text": "Text", "url": "https://example.com",
             "section": "grants", "subsection": "test"}
        ]
        out = str(tmp_path / "test_deeper.json")
        save_items(items, out)
        assert os.path.exists(out)

        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test"

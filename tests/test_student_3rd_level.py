"""
TDD tests for 3rd-level scraping on student.mirea.ru.

RED phase: All tests should FAIL before implementation in student_portal.py.

Tests cover:
1. 3RD_LEVEL_SECTIONS constant exists with correct values
2. _scrape_section follows links to 3rd-level subpages
3. Subpage content is extracted and added to results
4. Non-3rd-level sections remain at depth 2 (no 3rd-level recursion)
5. child_links cap is raised to 25 for 3rd-level sections
6. URL deduplication prevents revisiting
"""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper.student_portal import (
    SECTIONS,
    _scrape_section,
    THIRD_LEVEL_SECTIONS,
)


# ---------------------------------------------------------------------------
# 1. Test 3RD_LEVEL_SECTIONS constant
# ---------------------------------------------------------------------------

class TestThirdLevelSectionsConstant:
    """Verify 3RD_LEVEL_SECTIONS is defined with correct values."""

    def test_third_level_sections_exists(self):
        """3RD_LEVEL_SECTIONS must be importable."""
        from src.scraper.student_portal import THIRD_LEVEL_SECTIONS
        assert THIRD_LEVEL_SECTIONS is not None

    def test_third_level_sections_has_three_entries(self):
        """Must contain exactly 3 section keys."""
        from src.scraper.student_portal import THIRD_LEVEL_SECTIONS
        assert len(THIRD_LEVEL_SECTIONS) == 3

    def test_third_level_sections_values(self):
        """Must contain scholaship_support, psychological_service, student_association."""
        from src.scraper.student_portal import THIRD_LEVEL_SECTIONS
        expected = {"scholaship_support", "psychological_service", "student_association"}
        assert THIRD_LEVEL_SECTIONS == expected

    def test_third_level_sections_are_subsets_of_sections(self):
        """All THIRD_LEVEL_SECTIONS keys must exist in SECTIONS dict."""
        from src.scraper.student_portal import THIRD_LEVEL_SECTIONS
        for key in THIRD_LEVEL_SECTIONS:
            assert key in SECTIONS, (
                f"THIRD_LEVEL_SECTIONS key '{key}' not found in SECTIONS dict"
            )


# ---------------------------------------------------------------------------
# 2. Test _scrape_section follows 3rd-level subpages
# ---------------------------------------------------------------------------

class TestScrapeSectionThirdLevel:
    """Verify _scrape_section fetches 3rd-level subpages for designated sections."""

    def _make_html(self, title="Test Page", body_content="<p>Some content here that is long enough for extraction.</p>", links=None):
        """Helper to build HTML strings for mocking."""
        link_html = ""
        if links:
            for href, text in links:
                link_html += f'<a href="{href}">{text}</a>\n'
        return f"""
        <html><body>
        <h1>{title}</h1>
        <div class="content">
            {body_content}
            {link_html}
        </div>
        </body></html>
        """

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_scrape_section_fetches_subpages_for_scholaship_support(self, mock_sleep, mock_fetch):
        """For scholaship_support section, _scrape_section should follow child links to 3rd level."""
        from src.scraper.student_portal import _scrape_section, THIRD_LEVEL_SECTIONS

        # Section page HTML with links to subpages
        section_html = self._make_html(
            title="Стипендии",
            body_content="<p>Основная информация о стипендиях и выплатах студентам.</p>",
            links=[
                ("/scholaship_support/bank_card/", "Банковская карта"),
                ("/scholaship_support/scholarships/", "Стипендии"),
                ("/scholaship_support/social_payment/", "Социальные выплаты"),
            ],
        )

        # Subpage HTML (3rd level)
        subpage_html = self._make_html(
            title="Банковская карта",
            body_content="<p>Информация о банковских картах для получения стипендии.</p>",
        )

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "scholaship_support", SECTIONS["scholaship_support"], items, seen_urls)

        # Should have fetched section page + 3 subpages = 4 fetch calls total
        assert mock_fetch.call_count >= 4, (
            f"Expected >= 4 fetch calls (section + 3 subpages), got {mock_fetch.call_count}"
        )

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_scrape_section_fetches_subpages_for_psychological_service(self, mock_sleep, mock_fetch):
        """For psychological_service section, subpages should be fetched."""
        from src.scraper.student_portal import _scrape_section

        section_html = self._make_html(
            title="Психологическая служба",
            body_content="<p>Психологическая служба оказывает помощь студентам.</p>",
            links=[
                ("/psychological_service/consultancy/", "Консультирование"),
                ("/psychological_service/FAQ/", "Вопросы и ответы"),
            ],
        )

        subpage_html = self._make_html(
            title="Консультирование",
            body_content="<p>Индивидуальные и групповые консультации психолога.</p>",
        )

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "psychological_service", SECTIONS["psychological_service"], items, seen_urls)

        # Should have fetched section + 2 subpages = 3 calls
        assert mock_fetch.call_count >= 3

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_scrape_section_fetches_subpages_for_student_association(self, mock_sleep, mock_fetch):
        """For student_association section, subpages should be fetched."""
        from src.scraper.student_portal import _scrape_section

        section_html = self._make_html(
            title="Студенческие объединения",
            body_content="<p>Студенческие объединения РТУ МИРЭА.</p>",
            links=[
                ("/student_association/committee_students/", "Комитет студентов"),
                ("/student_association/student_groups/", "Студенческие отряды"),
                ("/student_association/student_union/", "Профсоюз"),
                ("/student_association/history-of-student-groups", "История"),
            ],
        )

        subpage_html = self._make_html(
            title="Комитет студентов",
            body_content="<p>Комитет студентов представляет интересы обучающихся.</p>",
        )

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "student_association", SECTIONS["student_association"], items, seen_urls)

        # Section + 4 subpages = 5 calls
        assert mock_fetch.call_count >= 5


# ---------------------------------------------------------------------------
# 3. Test subpage content is extracted into results
# ---------------------------------------------------------------------------

class TestSubpageContentExtraction:
    """Verify 3rd-level subpage content ends up in items list."""

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_subpage_content_in_items(self, mock_sleep, mock_fetch):
        """Content from 3rd-level subpages should appear in the items list."""
        from src.scraper.student_portal import _scrape_section

        section_html = """
        <html><body>
        <h1>Стипендии</h1>
        <div class="content">
            <p>Основная информация о стипендиях.</p>
            <a href="/scholaship_support/bank_card/">Банковская карта</a>
        </div>
        </body></html>
        """

        subpage_html = """
        <html><body>
        <h1>Банковская карта для стипендии</h1>
        <div class="content">
            <p>Для получения стипендии необходимо оформить банковскую карту МИР.</p>
            <p>Подробная информация о процедуре оформления и необходимых документах.</p>
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "scholaship_support", SECTIONS["scholaship_support"], items, seen_urls)

        # Should have items from both section page and subpage
        assert len(items) >= 2, f"Expected >= 2 items, got {len(items)}"

        # Subpage content should be present
        subpage_items = [item for item in items if "Банковская карта" in item.get("title", "")]
        assert len(subpage_items) >= 1, "Subpage content not found in items"

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_subpage_item_has_correct_schema(self, mock_sleep, mock_fetch):
        """Each 3rd-level item must have the standard schema keys."""
        from src.scraper.student_portal import _scrape_section

        section_html = """
        <html><body>
        <h1>Психологическая служба</h1>
        <div class="content">
            <p>Психологическая служба университета.</p>
            <a href="/psychological_service/consultancy/">Консультации</a>
        </div>
        </body></html>
        """

        subpage_html = """
        <html><body>
        <h1>Психологическое консультирование</h1>
        <div class="content">
            <p>Индивидуальные консультации проводятся по записи.</p>
            <p>Продолжительность — 50 минут.</p>
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "psychological_service", SECTIONS["psychological_service"], items, seen_urls)

        required_keys = {"title", "text", "url", "section", "subsection"}
        for item in items:
            assert required_keys.issubset(item.keys()), (
                f"Item missing keys: {required_keys - item.keys()}"
            )
            assert item["section"] == "student_portal"


# ---------------------------------------------------------------------------
# 4. Test non-3rd-level sections stay at depth 2
# ---------------------------------------------------------------------------

class TestNonThirdLevelDepth:
    """Verify non-THIRD_LEVEL_SECTIONS do NOT get 3rd-level scraping."""

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_non_third_level_section_cap_10(self, mock_sleep, mock_fetch):
        """Non-3rd-level sections should cap child_links at 10 (not 25)."""
        from src.scraper.student_portal import _scrape_section

        # Build section HTML with 15 child links
        links = [(f"/obshchezhitiya/dorm{i}/", f"Общежитие {i}") for i in range(15)]
        section_html = f"""
        <html><body>
        <h1>Общежития</h1>
        <div class="content">
            <p>Информация об общежитиях РТУ МИРЭА для студентов.</p>
            {''.join(f'<a href="{href}">{text}</a>' for href, text in links)}
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        # Return 200 for any child fetch
        child_resp = MagicMock(status_code=200)
        child_resp.text = "<html><body><h1>Child</h1><div class='content'><p>Content for the child page that is long enough.</p></div></body></html>"
        child_resp.encoding = "utf-8"

        responses = [section_resp] + [child_resp] * 15
        mock_fetch.side_effect = responses

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "obshchezhitiya", SECTIONS["obshchezhitiya"], items, seen_urls)

        # Should cap at 10 child links for non-3rd-level section
        # Total fetches = 1 (section) + 10 (children) = 11
        assert mock_fetch.call_count <= 11, (
            f"Non-3rd-level section should cap at 10 children, got {mock_fetch.call_count - 1} child fetches"
        )


# ---------------------------------------------------------------------------
# 5. Test child_links cap raised to 25 for 3rd-level sections
# ---------------------------------------------------------------------------

class TestChildLinksCap:
    """Verify child_links cap is raised to 25 for THIRD_LEVEL_SECTIONS."""

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_third_level_section_cap_25(self, mock_sleep, mock_fetch):
        """3rd-level sections should allow up to 25 child links."""
        from src.scraper.student_portal import _scrape_section

        # Build section HTML with 20 child links (more than 10 but less than 25)
        links = [(f"/scholaship_support/page{i}/", f"Страница {i}") for i in range(20)]
        section_html = f"""
        <html><body>
        <h1>Стипендии</h1>
        <div class="content">
            <p>Информация о стипендиальном обеспечении.</p>
            {''.join(f'<a href="{href}">{text}</a>' for href, text in links)}
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        child_resp = MagicMock(status_code=200)
        child_resp.text = "<html><body><h1>Sub</h1><div class='content'><p>Subpage content that is long enough for extraction.</p></div></body></html>"
        child_resp.encoding = "utf-8"

        responses = [section_resp] + [child_resp] * 20
        mock_fetch.side_effect = responses

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "scholaship_support", SECTIONS["scholaship_support"], items, seen_urls)

        # With cap at 25, all 20 children should be fetched
        # Total = 1 (section) + 20 (children) = 21
        assert mock_fetch.call_count >= 21, (
            f"3rd-level section should fetch up to 25 children, got {mock_fetch.call_count - 1} child fetches"
        )


# ---------------------------------------------------------------------------
# 6. Test URL deduplication
# ---------------------------------------------------------------------------

class TestURLDeduplication:
    """Verify no URL is visited twice."""

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_no_duplicate_url_fetches(self, mock_sleep, mock_fetch):
        """Same URL should not be fetched twice even if it appears in links."""
        from src.scraper.student_portal import _scrape_section

        section_html = """
        <html><body>
        <h1>Стипендии</h1>
        <div class="content">
            <p>Информация о стипендиях.</p>
            <a href="/scholaship_support/bank_card/">Банковская карта</a>
            <a href="/scholaship_support/bank_card/">Банковская карта (дубль)</a>
        </div>
        </body></html>
        """

        subpage_html = """
        <html><body>
        <h1>Банковская карта</h1>
        <div class="content">
            <p>Информация о банковской карте для стипендии.</p>
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "scholaship_support", SECTIONS["scholaship_support"], items, seen_urls)

        # Section page + 1 unique subpage = 2 fetch calls (NOT 3)
        assert mock_fetch.call_count == 2, (
            f"Duplicate URL should be skipped, expected 2 fetch calls, got {mock_fetch.call_count}"
        )

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_seen_urls_populated(self, mock_sleep, mock_fetch):
        """seen_urls set should be populated with all visited URLs."""
        from src.scraper.student_portal import _scrape_section

        section_html = """
        <html><body>
        <h1>Студенческие объединения</h1>
        <div class="content">
            <p>Студенческие объединения университета.</p>
            <a href="/student_association/committee_students/">Комитет</a>
        </div>
        </body></html>
        """

        subpage_html = """
        <html><body>
        <h1>Комитет студентов</h1>
        <div class="content">
            <p>Комитет студентов представляет интересы.</p>
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp = MagicMock(status_code=200)
        subpage_resp.text = subpage_html
        subpage_resp.encoding = "utf-8"

        mock_fetch.side_effect = [section_resp, subpage_resp, subpage_resp]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "student_association", SECTIONS["student_association"], items, seen_urls)

        # seen_urls should contain the section URL and the subpage URL
        assert SECTIONS["student_association"]["url"] in seen_urls
        assert any("committee_students" in u for u in seen_urls)


# ---------------------------------------------------------------------------
# 7. Test time.sleep(0.5) between subpage requests
# ---------------------------------------------------------------------------

class TestSubpageRateLimiting:
    """Verify time.sleep(0.5) is called between 3rd-level subpage fetches."""

    @patch("src.scraper.student_portal._fetch")
    @patch("src.scraper.student_portal.time.sleep")
    def test_sleep_called_between_subpages(self, mock_sleep, mock_fetch):
        """time.sleep should be called with 0.5 between 3rd-level subpage requests."""
        from src.scraper.student_portal import _scrape_section

        section_html = """
        <html><body>
        <h1>Психологическая служба</h1>
        <div class="content">
            <p>Психологическая служба.</p>
            <a href="/psychological_service/consultancy/">Консультации</a>
            <a href="/psychological_service/FAQ/">FAQ</a>
        </div>
        </body></html>
        """

        subpage_html_consultancy = """
        <html><body>
        <h1>Консультации</h1>
        <div class="content">
            <p>Индивидуальные и групповые консультации.</p>
            <a href="/psychological_service/consultancy/individual/">Индивидуальные</a>
        </div>
        </body></html>
        """

        subpage_html_faq = """
        <html><body>
        <h1>FAQ</h1>
        <div class="content">
            <p>Часто задаваемые вопросы о работе службы.</p>
        </div>
        </body></html>
        """

        subpage_html_3rd = """
        <html><body>
        <h1>Индивидуальные консультации</h1>
        <div class="content">
            <p>Подробная информация об индивидуальных консультациях психолога.</p>
        </div>
        </body></html>
        """

        section_resp = MagicMock(status_code=200)
        section_resp.text = section_html
        section_resp.encoding = "utf-8"

        subpage_resp_1 = MagicMock(status_code=200)
        subpage_resp_1.text = subpage_html_consultancy
        subpage_resp_1.encoding = "utf-8"

        subpage_resp_2 = MagicMock(status_code=200)
        subpage_resp_2.text = subpage_html_faq
        subpage_resp_2.encoding = "utf-8"

        subpage_resp_3rd = MagicMock(status_code=200)
        subpage_resp_3rd.text = subpage_html_3rd
        subpage_resp_3rd.encoding = "utf-8"

        mock_fetch.side_effect = [
            section_resp,
            subpage_resp_1,
            subpage_resp_3rd,
            subpage_resp_2,
        ]

        session = MagicMock()
        items = []
        seen_urls = set()

        _scrape_section(session, "psychological_service", SECTIONS["psychological_service"], items, seen_urls)

        # Check that 0.5 was passed to time.sleep at least once (for subpage delays)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert 0.5 in sleep_calls, (
            f"time.sleep(0.5) not found in sleep calls: {sleep_calls}"
        )

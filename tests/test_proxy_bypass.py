"""
TDD tests for proxy bypass in all 6 scrapers.

All tests MUST FAIL initially (RED phase) because the proxy fix has not been applied yet.
They should PASS after the fix is implemented in T3.
"""
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.education import _make_session as education_make_session
from src.scraper.deeper_scraper import _make_session as deeper_make_session
from src.scraper.student_portal import _make_session as student_make_session
from src.scraper.applicants import _is_reachable as applicants_is_reachable
from src.scraper.sveden import _is_reachable as sveden_is_reachable
from src.scraper.contacts import _is_reachable as contacts_is_reachable
from src.scraper.applicants import scrape_applicants
from src.scraper.sveden import scrape_sveden
from src.scraper.contacts import scrape_contacts


class TestMakeSessionProxyBypass:

    @pytest.mark.parametrize(
        "make_session_func,name",
        [
            (education_make_session, "education"),
            (deeper_make_session, "deeper_scraper"),
            (student_make_session, "student_portal"),
        ],
    )
    def test_make_session_trust_env_false(self, make_session_func, name):
        session = make_session_func()
        assert session.trust_env is False, (
            f"{name}: _make_session() returned session with trust_env={session.trust_env!r}, expected False"
        )


class TestIsReachableProxyBypass:

    @pytest.mark.parametrize(
        "is_reachable_func,patch_target,name",
        [
            (applicants_is_reachable, "src.scraper.applicants.requests.head", "applicants"),
            (sveden_is_reachable, "src.scraper.sveden.requests.head", "sveden"),
            (contacts_is_reachable, "src.scraper.contacts.requests.head", "contacts"),
        ],
    )
    def test_is_reachable_bypasses_proxy(self, is_reachable_func, patch_target, name):
        fake_response = MagicMock(status_code=200, ok=True)

        with patch(patch_target, return_value=fake_response) as mock_head:
            is_reachable_func("https://example.com")

        _, kwargs = mock_head.call_args
        assert "proxies" in kwargs, (
            f"{name}: _is_reachable() did not pass 'proxies' to requests.head()"
        )
        assert kwargs["proxies"] == {"http": None, "https": None}, (
            f"{name}: _is_reachable() passed proxies={kwargs['proxies']!r}, "
            f"expected {{'http': None, 'https': None}}"
        )


class TestScrapeSessionProxyBypass:

    @pytest.mark.parametrize(
        "scrape_func,patch_module,name",
        [
            (scrape_applicants, "src.scraper.applicants", "applicants"),
            (scrape_sveden, "src.scraper.sveden", "sveden"),
            (scrape_contacts, "src.scraper.contacts", "contacts"),
        ],
    )
    def test_scrape_creates_session_without_proxy(self, scrape_func, patch_module, name):
        mock_session_instance = MagicMock()

        with patch(f"{patch_module}.requests.Session") as MockSession:
            MockSession.return_value = mock_session_instance
            with patch(f"{patch_module}._is_reachable", return_value=True):
                with patch(f"{patch_module}._safe_get", return_value=None):
                    with patch(f"{patch_module}.time.sleep"):
                        scrape_func()

        assert mock_session_instance.trust_env is False, (
            f"{name}: scrape function must set session.trust_env = False. "
            f"Current: {mock_session_instance.trust_env!r}"
        )

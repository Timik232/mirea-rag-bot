"""
Tests for network-layer behaviour of all 6 scrapers.

Historical intent (T3 proxy bypass): scraper sessions must NOT use ambient
system proxies, and requests must reach mirea.ru despite its anti-bot.

Current architecture: sessions are curl_cffi with Chrome TLS impersonation
(python-requests TLS fingerprint is banned with 403 by mirea.ru anti-bot).
curl_cffi sessions never read ambient proxy env by default, which preserves
the original proxy-bypass guarantee.
"""
import os
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


class TestMakeSessionImpersonation:
    """_make_session() must return a curl_cffi chrome-impersonated session."""

    @pytest.mark.parametrize(
        "make_session_func,name",
        [
            (education_make_session, "education"),
            (deeper_make_session, "deeper_scraper"),
            (student_make_session, "student_portal"),
        ],
    )
    def test_make_session_is_curl_cffi(self, make_session_func, name):
        from curl_cffi.requests import Session as CffiSession

        session = make_session_func()
        assert isinstance(session, CffiSession), (
            f"{name}: _make_session() returned {type(session).__name__}, "
            f"expected curl_cffi Session (Chrome impersonation)"
        )


class TestIsReachableNoRaise:
    """_is_reachable() must never raise; on network error it returns False."""

    @pytest.mark.parametrize(
        "is_reachable_func,name",
        [
            (applicants_is_reachable, "applicants"),
            (sveden_is_reachable, "sveden"),
            (contacts_is_reachable, "contacts"),
        ],
    )
    def test_is_reachable_swallows_errors(self, is_reachable_func, name):
        with patch(
            "curl_cffi.requests.head", side_effect=RuntimeError("boom")
        ):
            assert is_reachable_func("https://example.com") is False, (
                f"{name}: _is_reachable() must return False on network error, not raise"
            )

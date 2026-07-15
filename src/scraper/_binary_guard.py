"""Binary content guard — prevents scrapers from ingesting PDFs, images,
and other non-HTML content as text.

Imported by all scraper modules. Centralises three layers of protection:
1. URL-level: skip known binary file extensions
2. Response-level: check Content-Type header
3. Content-level: detect high ratio of non-printable characters
"""

import re
from urllib.parse import urlparse

# Extensions that are NEVER HTML pages — skip before fetching
BINARY_EXTENSIONS = frozenset({
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".rtf", ".pages", ".key",
    # Archives
    ".zip", ".rar", ".7z", ".gz", ".tar", ".bz2",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".bmp", ".tiff", ".tif", ".psd",
    # Audio/Video
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".wav",
    ".ogg", ".webm", ".m4a", ".m4v",
    # Executables
    ".exe", ".msi", ".dmg", ".deb", ".rpm", ".apk",
    # Data
    ".csv", ".xml", ".json", ".yaml", ".yml",
    # Other
    ".js", ".css", ".woff", ".woff2", ".ttf", ".eot",
})

# Content-Type prefixes that are acceptable for HTML scraping
ACCEPTABLE_CONTENT_TYPES = frozenset({
    "text/html",
    "application/xhtml+xml",
    "text/plain",
})

# Threshold for binary content detection in text
# If > this fraction of characters are non-printable, it's binary garbage
_BINARY_CHAR_THRESHOLD = 0.10
# Texts longer than this are suspicious (real pages are rarely > 50k chars)
_MAX_TEXT_LENGTH = 100_000


def is_binary_url(url: str) -> bool:
    """Check if URL points to a known binary file type."""
    path = urlparse(url).path.lower()
    for ext in BINARY_EXTENSIONS:
        if path.endswith(ext):
            return True
    # Also catch upload/media directories that often host binaries
    if "/upload/" in path or "/uploads/" in path or "/media/" in path:
        # Check if it has an HTML-looking ending
        if not path.endswith(("/", ".html", ".htm", ".php", ".asp", ".aspx")):
            return True
    return False



def is_binary_response(resp) -> bool:
    """Check if HTTP response contains binary content (by Content-Type header)."""
    content_type = resp.headers.get("Content-Type", "").lower()
    for ct in ACCEPTABLE_CONTENT_TYPES:
        if ct in content_type:
            return False
    # If we got here, content-type is not HTML/text — likely binary
    return True


def has_binary_content(text: str, sample_size: int = 5000) -> bool:
    """Detect binary garbage in text by checking ratio of non-printable chars.

    Samples the first `sample_size` characters for speed.
    """
    if len(text) > _MAX_TEXT_LENGTH:
        return True

    sample = text[:sample_size]
    if not sample:
        return False

    non_printable = 0
    for ch in sample:
        # Allow: printable ASCII, common whitespace, Unicode letters/digits
        # Flag: control chars (except \n\r\t), replacement char U+FFFD
        if ch == "\ufffd":
            non_printable += 1
        elif ord(ch) < 32 and ch not in "\n\r\t":
            non_printable += 1

    ratio = non_printable / len(sample)
    return ratio > _BINARY_CHAR_THRESHOLD

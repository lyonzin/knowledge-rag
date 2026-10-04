"""Bounded PDF-text recognition and cleanup after native open failures."""

from __future__ import annotations

import re
import traceback
from pathlib import Path

# Text extraction is bounded independently from unrestricted native PDF reading.
MAX_EXTRACTED_PDF_BYTES = 4 * 1024 * 1024
_PDF_MAGIC = b"%PDF-"
_SIGNATURE_SCAN_BYTES = 1024 + len(_PDF_MAGIC) - 1
_INITIAL_PAGE = re.compile(rb"\A(?:\xef\xbb\xbf)?\[Page [1-9][0-9]*\][ \t]*\r?\n")
_PAGE_LINES = re.compile(r"(?m)^\[Page [1-9][0-9]*\][ \t]*\r?$")
_NON_WHITESPACE = re.compile(r"\S")
_FORBIDDEN_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _read_candidate(filepath: Path) -> bytes | None:
    """Reject oversized files before opening; cap reads despite concurrent growth."""
    size = filepath.stat().st_size
    if size > MAX_EXTRACTED_PDF_BYTES:
        return None
    with filepath.open("rb") as stream:
        header = stream.read(min(_SIGNATURE_SCAN_BYTES, size + 1))
        if _PDF_MAGIC in header or not _INITIAL_PAGE.match(header):
            return None
        payload = header + stream.read(size + 1 - len(header))
    return payload if len(payload) <= size else None


def _has_page_text(content: str) -> bool:
    """Require text between markers without copying or normalizing the content."""
    start = 0
    for marker in _PAGE_LINES.finditer(content):
        if _NON_WHITESPACE.search(content, start, marker.start()):
            return True
        start = marker.end()
    return _NON_WHITESPACE.search(content, start) is not None


def read_extracted_pdf_text(filepath: Path) -> tuple[str, dict[str, str | int]] | None:
    """Recognize an explicit UTF-8 extraction, never a fallback for broken PDF bytes.

    Accept an initial [Page N] line with optional UTF-8 BOM and tab/CR/LF in
    the text. Reject NUL, other C0/C1 controls and DEL. The 4 MiB bound applies
    only to extracted text; files with PDF magic keep the native parser path.
    Page counts describe retained markers, not the original PDF's total pages.
    """
    payload = _read_candidate(filepath)
    if payload is None:
        return None
    try:
        content = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return None
    if _FORBIDDEN_CONTROLS.search(content) or not _has_page_text(content):
        return None
    metadata: dict[str, str | int] = {
        "source_format": "pdf",
        "content_format": "extracted_text",
        "pages": sum(1 for _ in _PAGE_LINES.finditer(content)),
        "page_count_source": "markers",
    }
    return content, metadata


def _clear_native_frames(error: BaseException) -> None:
    """Release only PyMuPDF-owned frame locals, never caller or generator state."""
    for frame, _ in traceback.walk_tb(error.__traceback__):
        module = frame.f_globals.get("__name__", "")
        if not isinstance(module, str) or module.partition(".")[0] not in {"pymupdf", "fitz"}:
            continue
        try:
            frame.clear()
        except RuntimeError:
            # A running frame cannot be cleared; only completed native calls matter.
            continue


def release_exception_frames(error: BaseException, *, stop_at: BaseException | None = None) -> None:
    """Release native streams held by completed frames after PDF open failure.

    A failed PyMuPDF constructor keeps its stream in exception-frame locals,
    locking the file during Windows staging cleanup. Clear only native-module
    locals, preserving exceptions, locations and messages. Do not cross the
    exception active before open: its chain can retain unrelated suspended
    generators whose cleanup must stay under the caller's control.
    """
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is stop_at or id(current) in seen:
            continue
        seen.add(id(current))
        _clear_native_frames(current)
        pending.extend(exc for exc in (current.__cause__, current.__context__) if exc is not None)

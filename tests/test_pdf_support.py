"""PDF-text contract and native error-resource regressions."""

import io
import traceback
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mcp_server import ingestion, pdf_support
from mcp_server.file_transaction import staged_text_file
from mcp_server.ingestion import DocumentParser


@pytest.mark.parametrize(
    "payload",
    [
        b"[Page 1]\nFirst page\n\n[Page 3]\nLast nonempty page",
        b"\xef\xbb\xbf[Page 1]\r\n\tFirst page\r\n[Page 3]\r\nLast nonempty page\r\n",
        "[Page 1]\nAção 漢字\n[Page 3]\nUnicode preserved".encode(),
    ],
)
def test_explicit_extracted_pdf_retains_source_identity_and_marks_content(tmp_path, payload):
    source = tmp_path / "extracted source.PDF"
    source.write_bytes(payload)

    document = DocumentParser().parse_file(source)

    assert document is not None
    assert document.source == source
    assert document.format == ".pdf"
    assert document.content == payload.decode("utf-8-sig")
    assert document.metadata["source_format"] == "pdf"
    assert document.metadata["content_format"] == "extracted_text"
    assert document.metadata["pages"] == 2
    assert document.metadata["page_count_source"] == "markers"
    assert document.chunks


def test_explicit_extraction_does_not_require_native_pdf_dependency(tmp_path, monkeypatch):
    source = tmp_path / "extracted.pdf"
    source.write_text("[Page 2]\nAn earlier empty page was not retained.", encoding="utf-8")
    monkeypatch.setattr(ingestion, "HAS_PYMUPDF", False)

    document = DocumentParser().parse_file(source)

    assert document is not None and document.metadata["pages"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        b"Plain UTF-8 text without a page marker",
        b"[Page 0]\nInvalid page number",
        b"[Page 1] missing newline",
        b"[Page 1]\n \t\r\n",
        b"[Page 1]\n\n[Page 2]\n\n[Page 3]\n",
        b"\xef\xbb\xbf[Page 1]\r\n \t\r\n[Page 3]\r\n",
        b"[Page 1]\n" + b"x" * 2048 + b"\xff",
        b"[Page 1]\nText\x00",
        b"[Page 1]\nText\x1b",
        b"[Page 1]\nText\x7f",
        "[Page 1]\nText\x85".encode(),
        b"%PDF-1.7\n[Page 1]\nCorrupt binary PDF",
        b"[Page 1]\n%PDF-1.7 inside signature window",
        b"[Page 1]\n" + b" " * 1014 + b"%PDF-1.7",
    ],
)
def test_unrecognized_or_binary_content_never_uses_text_fallback(tmp_path, payload):
    source = tmp_path / "invalid.pdf"
    source.write_bytes(payload)

    assert pdf_support.read_extracted_pdf_text(source) is None


def test_oversized_file_is_rejected_before_any_read(tmp_path):
    source = tmp_path / "too-large.pdf"
    with source.open("wb") as stream:
        stream.truncate(pdf_support.MAX_EXTRACTED_PDF_BYTES + 1)

    with patch.object(Path, "open", side_effect=AssertionError("must reject before opening")):
        assert pdf_support.read_extracted_pdf_text(source) is None


def test_text_at_size_limit_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_support, "MAX_EXTRACTED_PDF_BYTES", 64)
    source = tmp_path / "at-limit.pdf"
    payload = b"[Page 1]\n" + b"x" * 55
    assert len(payload) == 64
    source.write_bytes(payload)

    parsed = pdf_support.read_extracted_pdf_text(source)

    assert parsed is not None and parsed[0] == payload.decode()


def test_file_growing_after_stat_is_rejected_with_bounded_reads(tmp_path, monkeypatch):
    source = tmp_path / "growing.pdf"
    initial = b"[Page 1]\nInitial text"
    source.write_bytes(initial)
    original_open = Path.open
    requests = []

    class RecordingStream(io.BufferedReader):
        def read(self, size=-1):
            requests.append(size)
            return super().read(size)

    def grow_then_open(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            with original_open(path, "ab") as stream:
                stream.write(b"x" * (pdf_support.MAX_EXTRACTED_PDF_BYTES + 1))
            return RecordingStream(io.FileIO(path, "rb"))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_then_open)

    assert pdf_support.read_extracted_pdf_text(source) is None
    assert all(size >= 0 for size in requests)
    assert sum(requests) <= len(initial) + 1


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("content", ["Invalid PDF without extraction markers", "%PDF-1.7\n\x00Corrupt trailer"])
def test_pdf_failure_cleans_staging_with_exception_retained(tmp_path, existing, content):
    pymupdf = pytest.importorskip("pymupdf")
    destination = tmp_path / "document.pdf"
    original = b"Original bytes must survive a failed update"
    if existing:
        destination.write_bytes(original)

    with pytest.raises(pymupdf.FileDataError) as caught:
        with staged_text_file(destination, content) as staged:
            DocumentParser().parse_file(staged.path)

    assert caught.value.__cause__ is not None
    assert "__init__" in [frame.name for frame in traceback.extract_tb(caught.value.__traceback__)]
    assert not list(tmp_path.glob(".rag-pending-*"))
    assert destination.read_bytes() == original if existing else not destination.exists()


def test_releasing_native_exception_locals_preserves_formatted_trace_and_cause(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    source = tmp_path / "invalid-native.pdf"
    source.write_bytes(b"%PDF-1.7\n\x00Corrupt trailer")

    with pytest.raises(pymupdf.FileDataError) as caught:
        pymupdf.open(source)
    error = caught.value
    cause = error.__cause__
    formatted = "".join(traceback.format_exception(error))

    pdf_support.release_exception_frames(error)
    source.unlink()

    assert error is caught.value and error.__cause__ is cause
    assert "".join(traceback.format_exception(error)) == formatted


def test_exception_chain_cycles_are_cleared_once_without_replacing_errors(monkeypatch):
    error, cause, context = ValueError("open"), RuntimeError("cause"), RuntimeError("context")
    error.__cause__, error.__context__ = cause, context
    cause.__context__, context.__cause__ = error, cause
    clear_frames = Mock()
    monkeypatch.setattr(pdf_support, "_clear_native_frames", clear_frames)

    pdf_support.release_exception_frames(error)

    assert clear_frames.call_count == 3
    assert error.__cause__ is cause and error.__context__ is context
    assert cause.__context__ is error and context.__cause__ is cause


def test_cleanup_does_not_traverse_preexisting_exception_chain(monkeypatch):
    native, prior, prior_cause = RuntimeError("native"), ValueError("prior"), ValueError("prior cause")
    native.__context__, prior.__cause__ = prior, prior_cause
    clear_frames = Mock()
    monkeypatch.setattr(pdf_support, "_clear_native_frames", clear_frames)

    pdf_support.release_exception_frames(native, stop_at=prior)

    clear_frames.assert_called_once_with(native)
    assert native.__context__ is prior and prior.__cause__ is prior_cause


@pytest.fixture
def suspended_error():
    events = []

    def prior_operation():
        try:
            try:
                raise ValueError("Synthetic prior operation")
            except ValueError as error:
                yield error
            events.append("resumed")
        finally:
            events.append("finally")

    generator = prior_operation()
    try:
        yield generator, next(generator), events
    finally:
        generator.close()


def test_failed_open_preserves_suspended_generator_in_prior_context(tmp_path, suspended_error):
    pymupdf = pytest.importorskip("pymupdf")
    source = tmp_path / "corrupt.pdf"
    source.write_bytes(b"%PDF-1.7\n\x00Corrupt trailer")
    generator, prior, events = suspended_error

    try:
        raise prior
    except ValueError:
        with pytest.raises(pymupdf.FileDataError) as caught:
            DocumentParser().parse_file(source)

    source.unlink()
    assert caught.value.__cause__ is not None
    assert events == [] and generator.gi_frame is not None
    generator.close()
    assert events == ["finally"]


def test_direct_cleanup_does_not_finalize_caller_generator(suspended_error):
    generator, error, events = suspended_error

    pdf_support.release_exception_frames(error)

    assert events == [] and generator.gi_frame is not None


def test_valid_binary_pdf_uses_native_extraction_without_exception_cleanup(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    source = tmp_path / "valid.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Valid native PDF text")
        document.save(source)
    cleanup = Mock(side_effect=AssertionError("success must not clear exception frames"))
    monkeypatch.setattr(pdf_support, "release_exception_frames", cleanup)

    parsed = DocumentParser().parse_file(source)
    source.unlink()

    assert parsed is not None and "Valid native PDF text" in parsed.content
    assert parsed.metadata["pages"] == 1
    assert "content_format" not in parsed.metadata
    cleanup.assert_not_called()

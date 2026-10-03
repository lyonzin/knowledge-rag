"""Local file replacement with an explicit rollback boundary for document CRUD."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class StagedTextFile:
    """Prepare beside the destination so rename works on every supported OS."""

    def __init__(self, destination: Path, content: str) -> None:
        self.destination = destination
        self.backup: Path | None = None
        self.published = False
        self.committed = False
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".rag-pending-",
            suffix=destination.suffix,
            delete=False,
        ) as stream:
            self.path = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                self.path.unlink(missing_ok=True)
                raise

    def publish(self) -> None:
        """Keep the prior bytes on disk until the index and metadata commit."""
        if self.destination.exists():
            with tempfile.NamedTemporaryFile(
                dir=self.destination.parent, prefix=".rag-backup-", delete=False
            ) as stream:
                backup = Path(stream.name)
            try:
                os.replace(self.destination, backup)
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
            self.backup = backup
        os.replace(self.path, self.destination)
        self.published = True

    def close(self) -> None:
        """Restore old bytes after failure; never delete a backup whose restore failed."""
        if not self.committed:
            if self.backup is not None:
                os.replace(self.backup, self.destination)
            elif self.published:
                self.destination.unlink(missing_ok=True)
        for temporary in (self.path, self.backup):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    # Cleanup must not turn an already committed write into a failure.
                    logger.warning("Could not remove transaction temporary file %s: %s", temporary, exc)


@contextmanager
def staged_text_file(destination: Path, content: str) -> Iterator[StagedTextFile]:
    """Stage, publish explicitly, and retain rollback until ``committed`` is true."""
    staged = StagedTextFile(destination, content)
    try:
        yield staged
    finally:
        staged.close()

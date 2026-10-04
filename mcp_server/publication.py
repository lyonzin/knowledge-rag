"""Coordinate short collection publication with concurrent in-flight readers."""

from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import wraps
from threading import Condition
from typing import Callable, Concatenate, Iterator, Optional, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class PublicationLock:
    """Allow parallel readers; drain them before publishing a replacement.

    Writers acquire the orchestrator mutation lock first. Readers must never
    block acquiring that mutation lock (BM25 lazy loading uses try-acquire).
    A waiting publisher blocks new readers so sustained traffic cannot starve
    the brief rename/metadata/retirement phase. Staging inference holds neither
    side of this barrier.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Lease the current collection and related state until a read finishes."""
        with self._condition:
            self._condition.wait_for(lambda: not self._writer and not self._waiting_writers)
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Publish only after every reader of the previous generation exits."""
        with self._condition:
            self._waiting_writers += 1
            try:
                self._condition.wait_for(lambda: not self._writer and not self._readers)
                self._writer = True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


def collection_reader(method: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Hold one read lease across collection, BM25, hydration and cache access."""

    @wraps(method)
    def guarded(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
        lock = cast(Optional[PublicationLock], getattr(self, "_publication_lock", None))
        with lock.read() if lock is not None else nullcontext():
            return method(self, *args, **kwargs)

    return guarded


def collection_publication(orchestrator: object) -> AbstractContextManager[None]:
    """Return the writer lease, retaining compatibility with minimal test doubles."""
    lock = cast(Optional[PublicationLock], getattr(orchestrator, "_publication_lock", None))
    return lock.write() if lock is not None else nullcontext()

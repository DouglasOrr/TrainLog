"""Friendly API for writing log files."""

from __future__ import annotations

import builtins
import contextlib
import datetime
import random
import time
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from . import io

Event = Dict[str, Any]
EventAnnotation = Callable[[Event], Optional[Event]]
Annotation = Callable[[Optional[Event]], Optional[EventAnnotation]]
LogLineScope = Callable[[Event], ContextManager[None]]


class Writer(Protocol):
    """Stream capable of writing general objects (e.g. io.JsonLinesIO)."""

    def write(self, obj: Event) -> None:
        """Write out a single event."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flush the underlying stream - write out the buffered data."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying stream."""
        raise NotImplementedError


@contextlib.contextmanager
def add_duration(event: Event, ndigits: int = 3) -> Iterator[None]:
    """[LogLineScope] Add the time elapsed during a `log.adding` scope."""
    start = time.time()
    yield
    event["duration"] = round(time.time() - start, ndigits)


def set_time(header: Optional[Event], ndigits: int = 3) -> EventAnnotation:
    """[Annotation] Set the header date & elapsed fields."""
    start = time.time()
    if header is not None:
        header["time"] = datetime.datetime.now().isoformat()
    return lambda event: dict(elapsed=round(time.time() - start, ndigits))


def set_id(header: Optional[Event]) -> None:
    """[Annotation] Set the ID field of the header."""
    if header is not None:
        header["id"] = random.Random().randint(0, 1 << 64)


KIND_HEADER = "header"
DEFAULT_ANNOTATE: Tuple[Annotation, ...] = (set_time, set_id)
DEFAULT_SCOPES: Tuple[LogLineScope, ...] = (add_duration,)


class LogLine:
    """A mutable builder for a single line within a log.

    For example:

        with log.adding(kind="eval") as line:
            line.set(loss=compute_loss())
    """

    def __init__(self, log: Log, scopes: Iterable[LogLineScope], event: Event):
        self.log = log
        self.scopes = scopes
        self.event = event
        self.stack = contextlib.ExitStack()
        for scope in self.scopes:
            self.stack.enter_context(scope(self.event))
        self.finished = False

    def __enter__(self) -> LogLine:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.add_to_log()

    def set(self, **kwargs: Any) -> None:
        """Set/reset event parameters in the log"""
        if self.finished:
            raise ValueError("Log line set() after being added to the log")
        self.event.update(kwargs)

    def add_to_log(self) -> None:
        """Add the scope to the underlying log."""
        if self.finished:
            raise ValueError("Trying to add a log line to the log multiple times")
        self.stack.close()
        self.log.add(**self.event)
        self.finished = True


class Log:
    """A general-purpose event log.

    For example:

        with logger.open("log.jsonl") as log:
            log.add(kind="step", loss=step_loss())
            log.add(kind="step", loss=step_loss())
            with log.adding(kind="eval") as line:
                line.set(loss=eval_loss())
    """

    def __init__(  # pylint:disable=dangerous-default-value
        self,
        writer: Writer,
        header: Event = {},
        annotate: Iterable[Annotation] = (),
        default_annotate: bool = True,
    ):
        self.writer = writer
        header_event = None if header is None else dict(kind=KIND_HEADER, **header)
        all_annotate = (DEFAULT_ANNOTATE if default_annotate else ()) + tuple(annotate)
        self.annotators = [
            annotator
            for annotator in (fn(header_event) for fn in all_annotate)
            if annotator is not None
        ]
        if header_event is not None:
            self.add(**header_event)

    def __enter__(self) -> Log:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying stream (the log cannot be used afterwards).

        Note that this is called automatically if the log is used in a context manager
        (recommended) and it is an error to close() twice.
        """
        self.writer.close()

    def add(self, kind: Optional[str] = None, **args: Any) -> None:
        """Immediately add a single event to the log.

        The key "kind" is optional, but recommended if the log contains different kinds
        of events, to aid info extraction.

        Values must be serializable by the current writer (e.g. JSON-serializable).

        Note that extra keys may be added by "Annotations", which are set up in the
        Log constructor.
        """
        line = args if kind is None else dict(kind=kind, **args)
        for annotator in self.annotators:
            out = annotator(line)
            if out is not None:
                line.update(out)
        self.writer.write(line)
        self.writer.flush()

    def adding(
        self,
        kind: Optional[str] = None,
        _scopes: Iterable[LogLineScope] = (),
        _default_scopes: bool = True,
        **args: Any,
    ) -> LogLine:
        """Start building an event to add to the log.

        The key "kind" is optional, but recommended if the log contains different kinds
        of events, to aid info extraction.

        We recommend using this as a context manager:

            with log.adding(kind="eval") as line:
                line.set(loss=eval_loss())

        Scopes allow automated logging of before-and-after state, for example the
        (built-in, automatically added) scope `add_duration`. Disable built-in scopes
        with `default_scopes=False`.
        """
        all_scopes = tuple(_scopes) + (DEFAULT_SCOPES if _default_scopes else ())
        return LogLine(
            self,
            scopes=all_scopes,
            event=args if kind is None else dict(kind=kind, **args),
        )


class JsonLinesFileLog(Log):
    """A standard log that uses a `trainlog.io.JsonLinesIO` writer to a local file."""

    def __init__(
        self,
        path: str,
        gzip_on_close: bool = True,
        dump_args: Optional[Dict[str, Any]] = None,
        **args: Any,
    ):
        """Create a JsonLines writer the the given local path.

        If `gzip_on_close` is True, after the log is closed, the file is compressed
        to `<path>.gz` and the original deleted.
        """
        super().__init__(
            io.JsonLinesIO[Event](builtins.open(path, "w"), dump_args=dump_args),
            **args,
        )
        self.path = path
        self.gzip_on_close = gzip_on_close

    def close(self) -> None:
        """Close the underlying stream, optionally converting to gzip."""
        super().close()
        if self.gzip_on_close:
            io.gzip(self.path)


def open(  # pylint:disable=redefined-builtin
    path: str,
    _gzip_on_close: bool = True,
    _annotate: Iterable[Annotation] = (),
    _default_annotate: bool = True,
    _add_header: bool = True,
    **header: Any,
) -> JsonLinesFileLog:
    """Open a logger writing to a local JsonLines file.

    This is just syntactic sugar; if customization is needed, directly using
    `JsonLinesFileLog` or `Log` may be preferable.
    """
    if header and not _add_header:
        raise ValueError(
            "Trying to add keys to a header with _add_header=False.\n"
            f"Either remove the header {header.keys()} or use open(_add_header=True)."
        )
    return JsonLinesFileLog(
        path,
        gzip_on_close=_gzip_on_close,
        annotate=_annotate,
        default_annotate=_default_annotate,
        header=header if _add_header else None,
    )

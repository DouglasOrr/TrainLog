"""Core IO abstractions - reading & writing JSON Lines (https://jsonlines.org/)."""

from __future__ import annotations

import gzip as gzip_
import json
import os
import typing
from types import TracebackType
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TextIO,
    Type,
    TypeVar,
)

T = TypeVar("T")


class JsonLinesIO(Generic[T]):
    """Reader/writer for JSON Lines files.

    See https://jsonlines.org/.

    Similar to TextIO, but writes "JSON-able" objects rather than strings.
    """

    stream: TextIO

    def __init__(self, stream: TextIO):
        self.stream = stream

    def __enter__(self) -> JsonLinesIO[T]:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[T]:
        return self.objects()

    def close(self) -> None:
        """Close the underlying text stream."""
        self.stream.close()

    def flush(self) -> None:
        """Flush the underlying text stream."""
        self.stream.flush()

    def write(self, obj: T, **dump_args: Any) -> None:
        """Write an object to the file, as a JSON entry on a single line."""
        dump_args.setdefault("separators", (",", ":"))
        json.dump(obj, self.stream, **dump_args)
        self.stream.write("\n")

    def read(self, **load_args: Any) -> T:
        """Read a single object from the file.

        Throws EOFError if there are no more JSON objects in the file.
        """
        line = self.stream.readline()
        if not line:
            raise EOFError(
                "Attempting to read JSON data past the end of stream", self.stream
            )
        return typing.cast(T, json.loads(line, **load_args))

    def objects(self) -> Iterator[T]:
        """An iterator over objects in the file."""
        try:
            while True:
                yield self.read()
        except EOFError:
            pass


def open_maybe_gzip(
    path: str,
    mode: str = "r",
    gzip: Optional[bool] = None,  # pylint: disable=redefined-outer-name
) -> TextIO:
    """Open a file, but use gzip.open if appropriate.

    gzip -- Treat the file as GZIP? If `None`, autodetect based on path extension.
    """
    if gzip or (gzip is None and os.path.splitext(path)[-1] in (".gz", ".gzip")):
        # Mode should default to text, for consistency with `open()`
        gzip_mode = mode if "b" in mode or "t" in mode else mode + "t"
        return typing.cast(TextIO, gzip_.open(path, gzip_mode))
    return typing.cast(TextIO, open(path, mode))


def read_jsonlines(path: str) -> Iterator[T]:
    """Read JSON Lines from a local filesystem path."""
    with JsonLinesIO[T](open_maybe_gzip(path)) as reader:
        yield from reader


def write_jsonlines(
    path: str, objects: Iterable[T], dump_args: Optional[Dict[str, Any]] = None
) -> None:
    """Write JSON Lines to a local filesystem path."""
    with JsonLinesIO[T](open_maybe_gzip(path, "w")) as writer:
        for obj in objects:
            writer.write(obj, **(dump_args or {}))


def gzip(
    path: str, extension: str = ".gz", delete: bool = True, chunk_size: int = 1024
) -> None:
    """Gzip a local file (by default deleting the original afterwards)."""
    assert extension, "cannot write the gzip to the file being read"
    with open(path, "rb") as srcf, gzip_.open(str(path) + extension, "wb") as destf:
        buffer = bytearray(chunk_size)
        while True:
            count = srcf.readinto(buffer)  # type: ignore
            if not count:
                break
            destf.write(buffer[:count])
    if delete:
        os.remove(path)

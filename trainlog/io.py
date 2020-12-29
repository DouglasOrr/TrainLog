"""Core IO abstractions - reading & writing JSON Lines (https://jsonlines.org/).

This module provides `JsonLinesIO`, a simplified stream reader/writer for JSON lines,
which also adds support for serializing/deserializing numpy arrays.
"""

from __future__ import annotations

import functools as ft
import gzip as gzip_
import json
import os
import typing
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TextIO,
    Tuple,
    Type,
    TypeVar,
)

T = TypeVar("T")

NUMPY_DICT_KEY = "__numpy_dict"


def numpy_to_dict(array: Any) -> Dict[str, Any]:
    """Convert a numpy array to a JSON-able dictionary, `numpy_from_dict` restores."""
    return {
        NUMPY_DICT_KEY: 0,
        "shape": list(array.shape),
        "dtype": array.dtype.base.name,
        "items": array.flatten().tolist(),
    }


def numpy_from_dict(dict_: Dict[str, Any]) -> Any:
    """Convert a JSON-able dictionary from `numpy_to_dict` back to a numpy array.

    This test identifies valid `dict`:

        if trainlog.io.NUMPY_DICT_KEY in dict_:
            array = numpy_from_dict(dict_)
    """
    import numpy as np  # type: ignore  # pylint: disable=import-outside-toplevel

    assert dict_[NUMPY_DICT_KEY] == 0
    shape = tuple(dict_["shape"])
    dtype = np.dtype(dict_["dtype"])
    items = dict_["items"]
    return np.array(items, dtype=dtype).reshape(shape)


class JSONEncoderWithNumpy(json.JSONEncoder):
    """Add numpy array support using `numpy_to_dict`."""

    def default(self, o: Any) -> Any:
        if type(o).__name__ == "ndarray":
            return numpy_to_dict(o)
        return super().default(o)


ObjectHook = Callable[[Dict[str, Any]], Any]
ObjectPairsHook = Callable[[Iterable[Tuple[str, Any]]], Any]


class JSONDecoderWithNumpy(json.JSONDecoder):
    """Add numpy array support using `numpy_from_dict`."""

    @staticmethod
    def create_object_hook(
        next_hook: Optional[ObjectHook], dict_: Dict[str, Any]
    ) -> Any:
        """Create a chained object_hook, which tries `numpy_from_dict` first."""
        if NUMPY_DICT_KEY in dict_:
            return numpy_from_dict(dict_)
        if next_hook is not None:
            return next_hook(dict_)
        return dict_

    @staticmethod
    def create_object_pairs_hook(
        next_hook: ObjectPairsHook, obj_pairs: Iterable[Tuple[str, Any]]
    ) -> Any:
        """Create a chained object_pairs_hook, which tries `numpy_from_dict` first."""
        dict_ = dict(obj_pairs)
        if NUMPY_DICT_KEY in dict_:
            return numpy_from_dict(dict_)
        return next_hook(obj_pairs)

    def __init__(
        self,
        *,
        object_hook: Optional[ObjectHook] = None,
        object_pairs_hook: Optional[ObjectPairsHook] = None,
        **args: Any
    ):
        object_pairs_hook = (
            ft.partial(self.create_object_pairs_hook, object_pairs_hook)
            if object_pairs_hook is not None
            else None
        )
        super().__init__(
            object_hook=ft.partial(self.create_object_hook, object_hook),
            object_pairs_hook=object_pairs_hook,
            **args
        )


class JsonLinesIO(Generic[T]):
    """Reader/writer for JSON Lines files.

    See https://jsonlines.org/.

    Similar to `TextIO`, but writes "JSON-able" objects rather than strings.
    """

    stream: TextIO
    dump_args: Dict[str, Any]
    load_args: Dict[str, Any]

    def __init__(
        self,
        stream: TextIO,
        dump_args: Optional[Dict[str, Any]] = None,
        load_args: Optional[Dict[str, Any]] = None,
    ):
        self.stream = stream
        self.dump_args = dump_args.copy() if dump_args else {}
        self.dump_args.setdefault("separators", (",", ":"))
        self.dump_args.setdefault("cls", JSONEncoderWithNumpy)
        self.load_args = load_args.copy() if load_args else {}
        self.load_args.setdefault("cls", JSONDecoderWithNumpy)

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

    def write(self, obj: T) -> None:
        """Write an object to the file, as a JSON entry on a single line."""
        json.dump(obj, self.stream, **self.dump_args)
        self.stream.write("\n")

    def read(self) -> T:
        """Read a single object from the file.

        Throws EOFError if there are no more JSON objects in the file.
        """
        line = self.stream.readline()
        if not line:
            raise EOFError(
                "Attempting to read JSON data past the end of stream", self.stream
            )
        return typing.cast(T, json.loads(line, **self.load_args))

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


def read_jsonlines(
    path: str, load_args: Optional[Dict[str, Any]] = None
) -> Iterator[T]:
    """Read JSON Lines from a local filesystem path."""
    with JsonLinesIO[T](open_maybe_gzip(path), load_args=load_args) as reader:
        yield from reader


def write_jsonlines(
    path: str, objects: Iterable[T], dump_args: Optional[Dict[str, Any]] = None
) -> None:
    """Write JSON Lines to a local filesystem path."""
    with JsonLinesIO[T](open_maybe_gzip(path, "w"), dump_args=dump_args) as writer:
        for obj in objects:
            writer.write(obj)


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

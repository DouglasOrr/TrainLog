"""Friendly APIs for preprocessing log files.

Note that it's simple to handle log files directly, using `io.read_jsonlines`.
This API adds a few tools for dealing with heterogenous event streams, before
handing them off to a tabular data processing library such as pandas.

For example, we might write the following to generate data that's easy to work
with in pandas:

    import trainlog.ops as O

    logs = trainlog.logs.glob("results/*.jsonl.gz")

    logs = logs.apply(
        O.header("learning_rate"),
        O.count_if("step"),
        O.when("valid", O.window("step", 100, O.reduce_mean("loss"), "train_loss")),
    )

    df = logs["valid"].to_pandas()
"""

from __future__ import annotations

import datetime
import glob as pyglob
import os
import typing
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Set, Union

from . import io, ops
from .ops import Event


class JsonLinesFile:
    """An event stream that is lazily read from a JSONlines file.

    Note that if the first event in the file has {"kind": "header"}, this class
    automatically adds a key {"metadata": {"path": ..., "created": ...,
    "modified": ...}}.
    """

    def __init__(self, path: str, load_args: Optional[Dict[str, Any]] = None):
        self.path = path
        self.load_args = load_args

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    def __iter__(self) -> Iterator[Event]:
        iterator: Iterator[Event] = iter(io.read_jsonlines(self.path, self.load_args))
        try:
            first_event = next(iterator)
            if first_event.get("kind") == "header":
                first_event["metadata"] = self.metadata
            yield first_event
            yield from iterator
        except StopIteration:
            pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """A dictionary of metadata about the file loaded."""
        return dict(
            path=self.path,
            created=datetime.datetime.fromtimestamp(
                os.path.getctime(self.path)
            ).isoformat(),
            modified=datetime.datetime.fromtimestamp(
                os.path.getmtime(self.path)
            ).isoformat(),
        )


def _events_repr(events: Iterable[Event]) -> str:
    """A string summary for tuples or lists that doesn't print all the contents."""
    if isinstance(events, (tuple, list)):
        return f"[{len(events)}]"
    return repr(events)


class Transform:
    """An event stream produced by transforming another stream.

    If we compare two alternatives:

        operation(events)
        Transform(events, operation)

    The main difference is that the second can be iterated multiple times (as long
    as `events` can.)
    """

    def __init__(self, events: Iterable[Event], operation: ops.BaseOperation):
        self.events = events
        self.operation = operation

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_events_repr(self.events)}, {self.operation!r})"

    def __iter__(self) -> Iterator[Event]:
        return self.operation(iter(self.events))


def convert_to_pandas(events: Iterable[Event]) -> Any:
    """A wrapper around `pandas.DataFrame.from_dict` that lazily imports pandas."""
    import pandas  # type: ignore  # pylint: disable=import-outside-toplevel

    return pandas.DataFrame.from_dict(events)


def get_header(events: Iterable[Event]) -> Optional[Event]:
    """Extract the header from the event stream.

    The header must be the first event, and have {"kind": "header"}.
    """
    first = next(iter(events), None)
    if first is None:
        return None
    if first.get("kind") != "header":
        return None
    return first


class Log:
    """A friendly API for manipulating a single log file.

    The analysis support here is quite basic, focussed around handling of ordered
    heterogenous events. We suggest performing further analysis and plotting using
    {pandas, numpy, scipy, matplotlib, seaborn, etc.}
    """

    def __init__(self, events: Iterable[Event]):
        self.events = events
        self.header = get_header(events)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_events_repr(self.events)})"

    def __getitem__(self, kind: str) -> Log:
        """Select events of a given kind from the log.

        Equivalent to `log.filter(kind)`.
        """
        if isinstance(kind, int):
            raise TypeError(
                "Cannot use a `Log` as an iterable - consider `Log.events` instead"
            )
        return self.filter(kind)

    @property
    def kinds(self) -> Set[Optional[str]]:
        """The set of {"kind": kind} from all events in the log."""
        return {typing.cast(Optional[str], event.get("kind")) for event in self.events}

    def cache(self) -> Log:
        """Create a log that is loaded into memory, for efficient multiple-traversal.

        Note that this does not change `self`, but returns a new cached Log.
        """
        if isinstance(self.events, (tuple, list)):
            return self
        return type(self)(tuple(self.events))

    def apply(self, *operations: ops.Operation) -> Log:
        """Create a transformed log view of this log.

        Note that the transformation will be executed whenever the log `events`
        are traversed.

        For example:

            log.apply(ops.count_if("step"))
        """
        return type(self)(Transform(self.events, ops.group(*operations)))

    def filter(self, predicate: ops.AutoPredicate) -> Log:
        """Create a filtered log view of this log."""
        return type(self)(Transform(self.events, ops.filter(predicate)))

    def to_pandas(self) -> Any:
        """Convert the log to a pandas DataFrame.

        It's normally easiest to do this for a single event kind at a time.
        For example:

            dfv = log["valid"].to_pandas()
            dfs = log["step"].to_pandas()
        """
        return convert_to_pandas(self.events)


class LogSet:
    """A friendly "batched" API for manipulating a set of log files."""

    def __init__(self, logs: Sequence[Log]):
        self.logs = logs

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self.logs)}])"

    def __getitem__(self, kind_or_index: Union[str, int]) -> Union[LogSet, Log]:
        """Either filter log events (str) or index a single log (int)."""
        if isinstance(kind_or_index, str):
            return self.filter(kind_or_index)
        return self.logs[kind_or_index]

    def __len__(self) -> int:
        return len(self.logs)

    def __iter__(self) -> Iterator[Log]:
        return iter(self.logs)

    @property
    def events(self) -> Iterator[Event]:
        """A concatenated stream of all events from all logs."""
        for log in self.logs:
            yield from log.events

    @property
    def kinds(self) -> Set[Optional[str]]:
        """The set of all {"kind": kind} from all events in all logs."""
        return {kind for log in self.logs for kind in log.kinds}

    def cache(self) -> LogSet:
        """Create a log set that is loaded into memory, for efficient multiple-traversal.

        Note that this does not change `self`, but returns a new cached LogSet.
        """
        return type(self)(tuple(log.cache() for log in self.logs))

    def apply(self, *operations: ops.Operation) -> LogSet:
        """Create a (per-event) transformed view of this set of logs.

        For example:

            logs.apply(ops.header("id"), ops.count_if("step"))
        """
        return type(self)(tuple(log.apply(*operations) for log in self.logs))

    def filter(self, predicate: ops.AutoPredicate) -> LogSet:
        """Create a (per-event) filtered view of this set of logs.

        For example:

            logs.filter("valid")
        """
        return type(self)(tuple(log.filter(predicate) for log in self.logs))

    def to_pandas(self) -> Any:
        """Convert the log to a pandas DataFrame.

        It's normally easiest to do this for a single event kind at a time.
        For example:

            logs["valid"].to_pandas()
        """
        return convert_to_pandas(self.events)


def load(path: str, load_args: Optional[Dict[str, Any]] = None) -> Log:
    """Load a single Log from a local JSONLines file (e.g. written by logger.Log).

    For example:

        log = load("results/log.jsonl.gz")
    """
    return Log(JsonLinesFile(path, load_args=load_args))


def glob(pattern: str, recursive: bool = False) -> LogSet:
    """Load all logs matched by a local filesystem glob.

    For example:

        logs = glob("results/**/*.jsonl*", recursive=True)
    """
    return LogSet(tuple(load(f) for f in pyglob.glob(pattern, recursive=recursive)))

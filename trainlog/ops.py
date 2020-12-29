"""Defines operations that transform log event sequences.

Includes a generic abstraction BaseOperation, for any transformation
of `Iterable[Event] -> Iterable[Event]`.

Most operations are derived from `Operation`, which defines an
operation that starts with an initial value and reduces over a log,
updating events as it goes.

For example, this operation can count the number of events that have
{"kind": "step"}:

    op = ops.Sum(ops.kind("step"), "step")
    events_out = op(events_in)

There is also a "functional" API that provides some convenient defaults
and conversions. Since `kind("...")` is such a common case, there is an
automatic conversion from string. Also, `kind()` provides a default name
for the output key, so the above `op` is equivalent to:

    op = ops.sum("step")
"""

import abc
import builtins
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

# pylint:disable=redefined-builtin

T = TypeVar("T")
Event = Dict[str, Any]

Mapping = Callable[[Event], Any]
Predicate = Callable[[Event], bool]
AutoPredicate = Union[str, Predicate]
ScalarFn = Callable[[Event], Union[bool, float]]
AutoScalarFn = Union[str, ScalarFn]

Reduction = Callable[[Iterable[Event]], Any]
BaseOperation = Callable[[Iterator[Event]], Iterator[Event]]

# Operations


class Operation(abc.ABC):
    """Base class for log denormalization operations.

    An operation is a small program that is run over a sequence of events from a log:

        value = operation.initial()
        for event in events:
            operation.update(event, value)
            value = operation.accumulate(event, value)
    """

    # pylint:disable=no-self-use

    def initial(self) -> Any:
        """[Optional] An initial value for the reduction."""
        return None

    def accumulate(
        self, event: Event, value: Any  # pylint:disable=unused-argument
    ) -> Any:
        """[Optional] A reducing function to generate a new value."""
        return None

    @abc.abstractmethod
    def update(self, event: Event, value: Any) -> None:
        """Applies the operation to update the given event.

        This operation is run before the event is `accumulate`d into `value`.

        The default implementation of `__call__` makes a shallow copy of `event`
        externally, so that it's safe to add or remove toplevel keys, however the
        `update()` implementation is responsible for deep copies, if required.

        Note that this may not be called for every event that accumulate() is called on
        (see `When`).
        """
        raise NotImplementedError

    def __call__(self, events: Iterator[Event]) -> Iterator[Event]:
        """Run the operation over an event stream, yielding a new stream."""
        # pylint:disable=assignment-from-none
        value = self.initial()
        for event in events:
            event = event.copy()
            self.update(event, value)
            yield event
            value = self.accumulate(event, value)


class Map(Operation):
    """Apply an elementwise mapping function."""

    def __init__(self, mapping: Mapping, name: str):
        self.mapping = mapping
        self.name = name

    def update(self, event: Event, value: None) -> None:
        event[self.name] = self.mapping(event)


class Copy(Operation):
    """Copy a value from a previous event."""

    def __init__(self, predicate: Predicate, src: str, dest: str):
        self.predicate = predicate
        self.src = src
        self.dest = dest

    def accumulate(self, event: Event, value: Any) -> Any:
        return event[self.src] if self.predicate(event) else value

    def update(self, event: Event, value: Any) -> None:
        if not self.predicate(event):
            event[self.dest] = value


class Sum(Operation):
    """Compute a scalar sum of previous events.

    E.g. counting events of a specific kind.
    """

    def __init__(self, scalarfn: ScalarFn, name: str):
        self.scalarfn = scalarfn
        self.name = name

    def initial(self) -> float:
        return 0

    def accumulate(self, event: Event, value: float) -> float:
        return value + self.scalarfn(event)

    def update(self, event: Event, value: float) -> None:
        event[self.name] = value


class Window(Operation):
    """Aggregate over a sliding window of preceding events matching a predicate."""

    def __init__(
        self, predicate: Predicate, size: Optional[int], reduction: Reduction, name: str
    ):
        self.predicate = predicate
        self.size = size
        self.reduction = reduction
        self.name = name

    def initial(self) -> List[Event]:
        return []

    def accumulate(self, event: Event, value: List[Event]) -> List[Event]:
        if self.predicate(event):
            value.append(event)
            if self.size is not None and len(value) > self.size:
                value = value[-self.size :]
        return value

    def update(self, event: Event, value: List[Event]) -> None:
        event[self.name] = self.reduction(value)


class Group(Operation):
    """Create a single operation that applies multiple operations in series."""

    def __init__(self, operations: Iterable[Operation]):
        self.operations = operations

    def initial(self) -> List[Any]:
        return [op.initial() for op in self.operations]

    def accumulate(self, event: Event, value: List[Any]) -> List[Any]:
        return [
            op.accumulate(event, opval) for op, opval in zip(self.operations, value)
        ]

    def update(self, event: Event, value: List[Any]) -> None:
        for op, opval in zip(self.operations, value):
            op.update(event, opval)


class When(Operation):
    """[Wrapper] Conditionally run update() for an operation.

    E.g. with `kind("step")` to only update step events
    """

    def __init__(self, predicate: Predicate, body: Operation):
        self.predicate = predicate
        self.body = body

    def initial(self) -> Any:
        return self.body.initial()

    def accumulate(self, event: Event, value: Any) -> Any:
        return self.body.accumulate(event, value)

    def update(self, event: Event, value: Any) -> None:
        if self.predicate(event):
            self.body.update(event, value)


class Duck(Operation):
    """Swallow key errors to make an operation "duck typed" on the event.

    Note: if the wrapped event raises a `KeyError`, it must leave `event`
    unchanged.
    """

    def __init__(self, body: Operation):
        self.body = body

    def initial(self) -> Any:
        return self.body.initial()

    def accumulate(self, event: Event, value: Any) -> Any:
        try:
            return self.body.accumulate(event, value)
        except KeyError:
            return value

    def update(self, event: Event, value: Any) -> None:
        try:
            self.body.update(event, value)
        except KeyError:
            pass


def _auto_name(
    name: Optional[str], fn: Callable[[Any], Any], prefix: str, context: str
) -> str:
    if name is not None:
        return name
    if fn.__name__ in {None, "<lambda>"}:
        raise ValueError(
            f"If `{context}()` using a lambda, you must provide a `name=...`"
        )
    return prefix + fn.__name__


def filter(predicate: AutoPredicate) -> BaseOperation:
    """An operation to filter an event stream by predicate or kind.

    For example:

        filter("valid")
        filter(lambda event: event["step"] >= 100)
    """
    pred = to_predicate(predicate)
    return lambda events: builtins.filter(pred, events)


def map(mapping: Mapping, name: Optional[str] = None) -> Operation:
    """An operation to apply a "pointwise" mapping function to events.

    The default `name` is `mapping.__name__`, unless None or "<lambda>".

    For example:

        map(lambda event: 1 - event["error_rate"], "accuracy")
    """
    return Map(mapping, _auto_name(name, mapping, prefix="", context="map"))


def copy(predicate: AutoPredicate, src: str, dest: Optional[str] = None) -> Operation:
    """An operation to copy a field from a previous event.

    The default `dest` is `src`.

    For example:

        copy("valid", "loss", "last_valid_loss")

    Equivalent to:

        value = None
        for event in events:
            if predicate(event):
                value = event[src]
            else:
                event[dest] = value
    """
    return Copy(to_predicate(predicate), src, dest or src)


def header(src: str, dest: Optional[str] = None) -> Operation:
    """An operation to copy a field from the header.

    Equivalent to `copy("header", src, dest)`.

    For example:

        header("id")
        header("learning_rate", "lr")
    """
    return Copy(kind("header"), src, dest or src)


def sum(scalarfn: AutoScalarFn, name: Optional[str] = None) -> Operation:
    """An operation to sum values from previous events.

    The default `name` is `sum_{scalarfn.__name__}`, unless None or "<lambda>".

    For example:

        sum(ops.get("examples"))
        sum(lambda event: event["examples"], "sum_examples")
    """
    scalarfn = to_predicate(scalarfn)
    return Sum(scalarfn, _auto_name(name, scalarfn, prefix="sum_", context="sum"))


def count(predicate: AutoPredicate, name: Optional[str] = None) -> Operation:
    """An operation to count occurrences of previous events.

    The default `name` is `predicate.__name__`, unless None or "<lambda>".

    For example:

        count("step")
        count(lambda event: event["loss"] > 10, "n_large_losses")
    """
    predicate = to_predicate(predicate)
    return Sum(predicate, _auto_name(name, predicate, prefix="", context="count"))


def window(
    predicate: AutoPredicate,
    size: Optional[int],
    reduction: Reduction,
    name: Optional[str] = None,
) -> Operation:
    """An operation to compute a statistic over a window of previous events.

    The default `name` is `reduction.__name__`, unless None or "<lambda>".

    For example:

        window("step", 10, ops.reduce_mean("loss"), "train_loss")
        window("step", 10,
            lambda events: np.mean([e["loss"] for e in events]),
            "train_loss")

    Equivalent to:

        previous = []
        for event in events:
            event[name] = reduction(previous)
            if predicate(event):
                previous = (previous + [event])[-size:]
    """
    return Window(
        to_predicate(predicate),
        size,
        reduction,
        _auto_name(name, reduction, prefix="", context="window"),
    )


def when(predicate: AutoPredicate, body: Operation) -> Operation:
    """An operation wrapper that only updates events when `predicate(event)`.

    For example:

        when("valid", ops.map(lambda event: 1 - event["error_rate"], "accuracy"))
    """
    return When(to_predicate(predicate), body)


def group(*operations: Operation) -> Operation:
    """An operation wrapper to apply multiple operations in a single pass.

    All `op.update()` are run first, followed by `op.accumulate()`.

    For example:

        group(ops.header("id"), ops.count("step"))
    """
    return operations[0] if len(operations) == 1 else Group(operations)


def duck(operation: Operation) -> Operation:
    """An operation wrapper to catch and ignore any KeyError from the operation.

    This can be used to "duck type" an operation & only apply it to events with
    the expected keys.

    For example:

        duck(ops.map(lambda event: 1 - event["error_rate"], "accuracy"))
    """
    return Duck(operation)


# Predicate/Scalarfn


def kind(value: str) -> Predicate:
    """A predicate to match events with `{"kind": value}`."""

    def predicate(event: Event) -> bool:
        return event.get("kind") == value

    predicate.__name__ = value
    return predicate


def to_predicate(
    kind_or_fn: Union[str, Callable[[Event], T]]
) -> Callable[[Event], Union[bool, T]]:
    """Automatic conversion from string to predicate `kind(kind_or_fn)`."""
    if isinstance(kind_or_fn, str):
        return kind(kind_or_fn)
    return kind_or_fn


def get(key: str, required: bool = False) -> ScalarFn:
    """A scalarfn for use with `sum` that gets a key from the event.

    If `required`, raise a KeyError if the key is missing (maybe useful with `duck`),
    otherwise returns `None`.

    Functionally equivalent to `operators.itemgetter(key)` when required is True,
    or `lambda e: e.get(key)` if required is False.

    Unlike lambdas or `operators`, this sets `__name__` so that there is a sensible
    default name for the generated event.

    For example:

        ops.sum(get("examples"))  # sums "examples" => "sum_examples"
    """

    def scalarfn(event: Event) -> Any:
        return event[key] if required else event.get(key)

    scalarfn.__name__ = key
    return scalarfn


# Reductions


def reduce_mean(key: str) -> Callable[[Iterable[Event]], Optional[float]]:
    """A reduction for use with `window`, that computes the mean of a scalar key.

    If passed an empty list, returns `None`.
    """

    def reduce(events: Iterable[Event]) -> Optional[float]:
        total = 0
        n = 0
        for event in events:
            total += event[key]
            n += 1
        return total / n if n else None

    reduce.__name__ = f"mean_{key}"
    return reduce

import pytest
from pytest import approx

from .. import ops


def _run(operation, *events):
    return list(operation(events))


def test_filter():
    assert _run(
        ops.filter("step"),
        dict(id=0, kind="header"),
        dict(id=1, kind="step"),
        dict(id=2, kind="valid"),
        dict(id=3, kind="step"),
    ) == [
        dict(id=1, kind="step"),
        dict(id=3, kind="step"),
    ]


def test_map():
    assert _run(
        ops.map(lambda event: 1 - event["error_rate"], "accuracy"),
        dict(id=0, error_rate=0.2),
        dict(id=1, error_rate=0.9),
    ) == [
        dict(id=0, error_rate=0.2, accuracy=approx(0.8)),
        dict(id=1, error_rate=0.9, accuracy=approx(0.1)),
    ]


def test_copy():
    assert _run(
        ops.copy("tick", "name", "last_tick"),
        dict(kind="step", name=0),
        dict(kind="tick", name="a"),
        dict(kind="tick", name="b"),
        dict(kind="step", name=1),
        dict(kind="step", name=2),
    ) == [
        dict(kind="step", name=0, last_tick=None),
        dict(kind="tick", name="a"),
        dict(kind="tick", name="b"),
        dict(kind="step", name=1, last_tick="b"),
        dict(kind="step", name=2, last_tick="b"),
    ]


def test_header():
    assert _run(
        ops.header("time"),
        dict(kind="header", time="noon"),
        dict(kind="valid"),
    ) == [
        dict(kind="header", time="noon"),
        dict(kind="valid", time="noon"),
    ]


def test_sum():
    assert _run(
        ops.sum(lambda x: x["examples"], "total_examples"),
        dict(id=0, examples=10),
        dict(id=1, examples=9),
        dict(id=2, examples=8),
    ) == [
        dict(id=0, examples=10, total_examples=0),
        dict(id=1, examples=9, total_examples=10),
        dict(id=2, examples=8, total_examples=19),
    ]

    assert _run(
        ops.sum(ops.get("examples")),
        dict(id=0, examples=10),
        dict(id=1, examples=9),
        dict(id=2, examples=8),
    ) == [
        dict(id=0, examples=10, sum_examples=0),
        dict(id=1, examples=9, sum_examples=10),
        dict(id=2, examples=8, sum_examples=19),
    ]

    with pytest.raises(ValueError):
        ops.sum(lambda x: x["examples"])


def test_count_if():
    assert _run(
        ops.count_if(ops.kind("step")),
        dict(kind="header"),
        dict(kind="step"),
        dict(kind="valid"),
        dict(kind="step"),
        dict(kind="valid"),
    ) == [
        dict(kind="header", step=0),
        dict(kind="step", step=0),
        dict(kind="valid", step=1),
        dict(kind="step", step=1),
        dict(kind="valid", step=2),
    ]


def test_window():
    assert _run(
        ops.window(ops.kind("step"), 2, ops.reduce_mean("loss")),
        dict(id=0, kind="step", loss=10),
        dict(id=1, kind="step", loss=20),
        dict(id=2, kind="step", loss=30),
        dict(id=3, kind="valid", loss=0),
        dict(id=4, kind="step", loss=40),
        dict(id=5, kind="valid", loss=0),
    ) == [
        dict(id=0, kind="step", loss=10, mean_loss=None),
        dict(id=1, kind="step", loss=20, mean_loss=10),
        dict(id=2, kind="step", loss=30, mean_loss=15),
        dict(id=3, kind="valid", loss=0, mean_loss=25),
        dict(id=4, kind="step", loss=40, mean_loss=25),
        dict(id=5, kind="valid", loss=0, mean_loss=35),
    ]


def accuracy(event):
    return 1 - event["error_rate"]


def test_group():
    assert _run(
        ops.group(ops.count_if("step"), ops.map(accuracy)),
        dict(kind="step", error_rate=0.9),
        dict(kind="valid", error_rate=0.8),
        dict(kind="step", error_rate=0.5),
        dict(kind="step", error_rate=0.4),
    ) == [
        dict(kind="step", error_rate=0.9, step=0, accuracy=approx(0.1)),
        dict(kind="valid", error_rate=0.8, step=1, accuracy=approx(0.2)),
        dict(kind="step", error_rate=0.5, step=1, accuracy=approx(0.5)),
        dict(kind="step", error_rate=0.4, step=2, accuracy=approx(0.6)),
    ]


def test_when():
    assert _run(
        ops.when("valid", ops.map(lambda _: True, "tag")),
        dict(kind="step", id=0),
        dict(kind="valid", id=1),
    ) == [
        dict(kind="step", id=0),
        dict(kind="valid", id=1, tag=True),
    ]


def test_duck():
    assert _run(
        ops.duck(ops.map(accuracy)),
        dict(kind="step", loss=10),
        dict(kind="valid", loss=9, error_rate=0.8),
    ) == [
        dict(kind="step", loss=10),
        dict(kind="valid", loss=9, error_rate=0.8, accuracy=approx(0.2)),
    ]

    assert _run(
        ops.duck(ops.sum(ops.get("n", required=True))),
        dict(kind="step", n=10),
        dict(kind="valid"),
        dict(kind="step", n=5),
        dict(kind="valid"),
    ) == [
        dict(kind="step", n=10, sum_n=0),
        dict(kind="valid", sum_n=10),
        dict(kind="step", n=5, sum_n=10),
        dict(kind="valid", sum_n=15),
    ]

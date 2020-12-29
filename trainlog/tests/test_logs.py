import re

import numpy as np  # type: ignore
import pytest

from .. import io, logs, ops


def test_jsonlines_file(tmp_path):
    path = tmp_path / "log.jsonl"
    io.write_jsonlines(
        path,
        [
            dict(kind="header", custom=123),
            dict(kind="step", loss=100),
        ],
    )

    log = logs.JsonLinesFile(path)

    assert path.absolute().name in str(log)

    events = list(log)
    assert events[0]["kind"] == "header"
    assert events[0]["custom"] == 123
    assert events[0]["metadata"]["path"] == path.absolute()
    assert re.search(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", events[0]["metadata"]["modified"]
    )
    assert events[1] == dict(kind="step", loss=100)


def test_jsonlines_file_empty(tmp_path):
    with open(tmp_path / "log.jsonl", "w"):
        pass

    log = logs.JsonLinesFile(tmp_path / "log.jsonl")
    assert list(log) == []


def _assert_array_equal(left, right):
    np.testing.assert_equal(left, right)
    assert left.dtype == right.dtype


def test_list_to_array():
    assert logs.list_to_array([]).shape == (0,)
    _assert_array_equal(
        logs.list_to_array([1, 2, 3]), np.array([1, 2, 3], dtype=np.int)
    )
    _assert_array_equal(
        logs.list_to_array([1, None, 2]), np.array([1, np.nan, 2], dtype=np.float)
    )
    _assert_array_equal(
        logs.list_to_array([1, None, []]), np.array([1, None, []], dtype=np.object)
    )
    objarray = logs.list_to_array([[1, 2], [3, 4], [5, 6]])
    assert objarray.shape == (3,)
    assert objarray.dtype == np.object


def test_log():
    log = logs.Log(
        (
            dict(kind="header", lr=0.01, time="distant past"),
            dict(kind="step", loss=10),
            dict(kind="step", loss=5),
            dict(kind="valid", loss=6, error_rate=0.1),
        )
    )
    assert repr(log) == "Log([4])"
    assert log.header == dict(kind="header", lr=0.01, time="distant past")
    assert log.kinds == {"header", "step", "valid"}
    assert log.cache() is log
    assert list(log["step"].events) == [
        dict(kind="step", loss=10),
        dict(kind="step", loss=5),
    ]


def test_log_transform():
    log = logs.Log(
        (
            dict(kind="header", lr=0.01, time="distant past"),
            dict(kind="step", loss=10),
            dict(kind="step", loss=5),
            dict(kind="valid", loss=6, error_rate=0.1),
        )
    )
    tlog = log.apply(ops.header("lr"))
    assert "[4]" in repr(tlog)
    assert list(tlog.events) == [
        dict(kind="header", lr=0.01, time="distant past"),
        dict(kind="step", loss=10, lr=0.01),
        dict(kind="step", loss=5, lr=0.01),
        dict(kind="valid", loss=6, error_rate=0.1, lr=0.01),
    ]

    df = tlog["step"].to_pandas()
    assert set(df.columns) == {"kind", "loss", "lr"}
    np.testing.assert_equal(df.loss.array, [10, 5])


def test_log_columns():
    # pylint: disable=no-member
    log = logs.Log(
        (
            dict(kind="step", loss=10),
            dict(kind="step", loss=9),
            dict(kind="valid", loss=100, error_rate=0.1),
            dict(kind="step", loss=8),
        )
    )

    unordered_columns = log.to_columns()
    assert len(unordered_columns) == 3
    assert "error_rate" in repr(unordered_columns)
    assert unordered_columns.kind.dtype.kind == "U"  # type:ignore[attr-defined]
    np.testing.assert_equal(
        unordered_columns.loss, [10, 9, 100, 8]  # type:ignore[attr-defined]
    )
    np.testing.assert_equal(
        unordered_columns["error_rate"], [np.nan, np.nan, 0.1, np.nan]
    )
    with pytest.raises(ValueError):
        unordered_columns[0]  # pylint:disable=pointless-statement
    assert set(unordered_columns.to_pandas().columns) == {"kind", "loss", "error_rate"}

    ordered_columns = log.to_columns("kind", "loss")
    assert len(ordered_columns) == 2
    assert repr(ordered_columns) == "Columns('kind', 'loss')"
    np.testing.assert_equal(ordered_columns[1], [10, 9, 100, 8])
    np.testing.assert_equal(ordered_columns["loss"], [10, 9, 100, 8])

    assert tuple(ordered_columns.to_pandas().columns) == ("kind", "loss")


def test_log_no_header():
    assert logs.Log([]).header is None
    assert logs.Log([dict(kind="step")]).header is None
    assert logs.Log([dict(loss=10)]).header is None


def test_log_not_an_iterable():
    with pytest.raises(TypeError):
        list(logs.Log([dict(kind="step")]))  # type: ignore[call-overload]


def test_log_set():
    first = logs.Log((dict(kind="header", name="first"), dict(kind="eval", result=90)))
    second = logs.Log(
        (
            dict(kind="header", name="second"),
            dict(kind="step", loss=9.2),
            dict(kind="eval", result=93),
        )
    )
    logset = logs.LogSet((first, second)).cache()
    assert repr(logset) == "LogSet([2])"
    assert len(logset) == 2
    assert list(logset) == [first, second]
    assert logset[0] is first
    assert logset[1] is second
    assert logset.kinds == {"header", "eval", "step"}
    assert dict(kind="step", loss=9.2) in logset.events

    tlogset = logset.apply(ops.header("name"))["eval"]
    assert list(tlogset.events) == [
        dict(kind="eval", result=90, name="first"),
        dict(kind="eval", result=93, name="second"),
    ]
    np.testing.assert_equal(tlogset.to_pandas().result.array, [90, 93])


def test_load_and_glob(tmp_path):
    io.write_jsonlines(
        tmp_path / "first.jsonl",
        [
            dict(kind="header", name="first"),
            dict(kind="eval", result=90),
        ],
    )
    io.write_jsonlines(
        tmp_path / "second.jsonl.gz",
        [
            dict(kind="header", name="second"),
            dict(kind="eval", result=93),
        ],
    )

    log = logs.open(tmp_path / "first.jsonl")
    assert log.header is not None
    assert log.header["name"] == "first"

    logset = logs.glob((tmp_path / "*.jsonl*").as_posix()).cache()
    assert len(logset) == 2
    assert sorted(
        logset.apply(ops.header("name"))["eval"].events, key=lambda x: x["result"]
    ) == [
        dict(kind="eval", result=90, name="first"),
        dict(kind="eval", result=93, name="second"),
    ]

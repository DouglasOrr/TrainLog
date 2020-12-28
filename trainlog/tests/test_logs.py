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

    log = logs.load(tmp_path / "first.jsonl")
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

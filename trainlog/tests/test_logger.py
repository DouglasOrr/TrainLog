import contextlib
import random
import re
import time
import unittest.mock as um
from typing import List

import pytest

from .. import io, logger


def test_add_duration():
    event: logger.Event = {}
    with logger.add_duration(event):
        time.sleep(0.01)
    assert event["duration"] > 0


def test_set_time():
    header: logger.Event = {}
    func = logger.set_time(header)
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", header["time"])

    fields = func({})
    assert fields is not None and fields["elapsed"] >= 0

    fields = logger.set_time(None)({})
    assert fields is not None and fields["elapsed"] >= 0


def test_set_id():
    header: logger.Event = {}
    random.seed(123)
    logger.set_id(header)
    assert isinstance(header["id"], int)

    header2: logger.Event = {}
    random.seed(123)  # set_id should not respect the global random seed
    logger.set_id(header2)
    assert (
        header2["id"] != header["id"]
    ), "OK, this can fail randomly - but the chance should be extremely low"


def _test_set_id(header):
    if header is not None:
        header["id"] = "TEST_ID"


def _test_set_time(header):
    if header is not None:
        header["time"] = "TEST_TIME"
    counter = 0

    def handler(event):
        nonlocal counter
        event["elapsed"] = counter
        counter += 1

    return handler


@contextlib.contextmanager
def _test_add_duration(event):
    yield
    event["duration"] = "TEST_DURATION"


def test_log():
    writer = um.Mock()

    with logger.Log(
        writer,
        dict(learning_rate=0.01),
        annotate=(_test_set_id, _test_set_time),
        default_annotate=False,
    ) as log:
        log.add(kind="step", loss=5.0)
        with log.adding(
            "eval",
            partition="valid",
            _scopes=(_test_add_duration,),
            _default_scopes=False,
        ) as line:
            line.set(loss=4.5)
            line.set(error_rate=0.9)

    writer.write.assert_has_calls(
        [
            um.call(
                dict(
                    kind="header",
                    learning_rate=0.01,
                    id="TEST_ID",
                    time="TEST_TIME",
                    elapsed=0,
                )
            ),
            um.call(dict(kind="step", loss=5.0, elapsed=1)),
            um.call(
                dict(
                    kind="eval",
                    partition="valid",
                    loss=4.5,
                    error_rate=0.9,
                    duration="TEST_DURATION",
                    elapsed=2,
                )
            ),
        ]
    )
    writer.close.assert_called_once()


def test_log_errors():
    with logger.Log(um.Mock()) as log:
        with log.adding() as line:
            pass
        with pytest.raises(ValueError):
            line.set(foo="bar")
        with pytest.raises(ValueError):
            line.add_to_log()


def _assert_has(dict_, **mappings):
    for key, value in mappings.items():
        assert dict_[key] == value


def test_file_log(tmp_path):
    path = tmp_path / "test.jsonl"

    with logger.open(path, name="my_test") as log:
        log.add(kind="step", loss=5.0)
        log.add(kind="step", loss=4.5)
        with log.adding(kind="eval") as line:
            line.set(loss=4.7)

    events: List[logger.Event] = list(io.read_jsonlines(tmp_path / "test.jsonl.gz"))
    _assert_has(events[0], kind="header", name="my_test")
    _assert_has(events[1], kind="step", loss=5.0)
    _assert_has(events[2], kind="step", loss=4.5)
    _assert_has(events[3], kind="eval", loss=4.7)
    assert len(events) == 4


def test_file_log_no_header_no_gzip(tmp_path):
    path = tmp_path / "test.jsonl"
    with logger.open(path, _add_header=False, _gzip_on_close=False) as log:
        log.add(kind="step", loss=5.0)
    events: List[logger.Event] = list(io.read_jsonlines(tmp_path / "test.jsonl"))
    _assert_has(events[0], kind="step", loss=5.0)
    assert len(events) == 1


def test_file_log_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    with pytest.raises(ValueError) as error:
        logger.open(path, custom_name="my_test", _add_header=False)
    assert "custom_name" in str(error.value)
    assert not path.exists()

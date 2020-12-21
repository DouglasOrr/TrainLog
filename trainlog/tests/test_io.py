import io as pyio
from typing import Any

import pytest

from .. import io  # pylint:disable=reimported


def test_jsonlines():
    stream = pyio.StringIO()

    json = io.JsonLinesIO[Any](stream)
    json.write(dict(a=1, b="2"))
    json.write(None)
    json.write(dict(a=10, b="20"))

    json.stream.seek(0)
    assert json.read() == dict(a=1, b="2")
    assert json.read() is None
    assert json.read() == dict(a=10, b="20")
    with pytest.raises(EOFError):
        json.read()

    json.stream.seek(0)
    assert list(json) == [dict(a=1, b="2"), None, dict(a=10, b="20")]


def test_jsonlines_context():
    stream = pyio.StringIO()
    with io.JsonLinesIO[Any](stream) as json:
        json.write("intentionally blank")
        assert stream.getvalue() == '"intentionally blank"\n'
    assert stream.closed


def test_read_write_jsonlines(tmp_path):
    items = [dict(a=1, b=None), dict(c=True)]
    io.write_jsonlines(tmp_path / "test.jsonl", items)
    assert list(io.read_jsonlines(tmp_path / "test.jsonl")) == items

import gzip
import io as pyio
import os
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

    for filename in ["test.jsonl", "test.jsonl.gz"]:
        io.write_jsonlines(tmp_path / filename, items)
        assert list(io.read_jsonlines(tmp_path / filename)) == items


def test_gzip(tmp_path):
    original = tmp_path / "original.txt"
    with open(original, "w") as f:
        f.write("one\ntwo\n")

    io.gzip(original)

    assert not os.path.isfile(original)
    with gzip.open(tmp_path / "original.txt.gz", "rt") as f:
        assert list(f) == ["one\n", "two\n"]


def test_gzip_custom(tmp_path):
    original = tmp_path / "original.txt"
    with open(original, "w") as f:
        f.write("one\ntwo\n")

    io.gzip(original, extension=".gzip", delete=False, chunk_size=7)

    assert os.path.isfile(original)
    with gzip.open(tmp_path / "original.txt.gzip", "rt") as f:
        assert list(f) == ["one\n", "two\n"]

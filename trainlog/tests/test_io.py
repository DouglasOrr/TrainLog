import gzip
import io as pyio
import os
from typing import Any

import numpy as np  # type: ignore
import pytest

from .. import io  # pylint:disable=reimported


def test_jsonlinesio():
    stream = pyio.StringIO()

    json = io.JsonLinesIO[Any](stream)
    json.write(dict(a=1, b="2"))
    json.write(None)
    json.write(dict(a=10, b="20"))
    json.flush()

    json.stream.seek(0)
    assert json.read() == dict(a=1, b="2")
    assert json.read() is None
    assert json.read() == dict(a=10, b="20")
    with pytest.raises(EOFError):
        json.read()

    json.stream.seek(0)
    assert list(json) == [dict(a=1, b="2"), None, dict(a=10, b="20")]


def _loads(line: str, **args: Any) -> Any:
    json = io.JsonLinesIO[Any](pyio.StringIO(line), **args)
    return json.read()


def _dumps(obj: Any, **args: Any) -> str:
    stream = pyio.StringIO()
    json = io.JsonLinesIO[Any](stream, **args)
    json.write(obj)
    return stream.getvalue()


class CustomType:
    """Just for testing."""

    def __init__(self, **args):
        self.__dict__.update(args)

    @classmethod
    def to_json(cls, obj):
        assert isinstance(obj, cls)
        return dict(obj.__dict__, type="custom")

    @classmethod
    def from_json(cls, dict_):
        assert dict_["type"] == "custom"
        return cls(**{k: v for k, v in dict_.items() if k != "type"})


def test_jsonlinesio_dump_args():
    assert (
        _dumps(dict(c=3, a=1, b=2), dump_args=dict(sort_keys=True))
        == '{"a":1,"b":2,"c":3}\n'
    ), "sort_keys"

    assert (
        _dumps(dict(c=3, a=1, b=2), dump_args=dict(separators=(", ", ": ")))
        == '{"c": 3, "a": 1, "b": 2}\n'
    ), "separators"


def test_jsonlinesio_custom_hooks():
    original = CustomType(a=100, b=200)
    reloaded = _loads(
        _dumps(original, dump_args=dict(default=CustomType.to_json)),
        load_args=dict(object_hook=CustomType.from_json),
    )
    assert isinstance(reloaded, CustomType)
    assert reloaded.__dict__ == original.__dict__

    assert _loads('{"x": 1, "y": 2}', load_args=dict(object_pairs_hook=tuple)) == (
        ("x", 1),
        ("y", 2),
    ), "object_pairs_hook"


def _assert_array_equal(left, right):
    np.testing.assert_equal(left, right)
    assert left.dtype == right.dtype


def test_jsonlinesio_numpy():
    original = np.arange(10).reshape((2, 5)).astype(np.int64)
    reloaded = _loads(_dumps(original))
    _assert_array_equal(original, reloaded)


def test_jsonlinesio_numpy_scalar():
    original = np.array(123.4, dtype=np.float64)
    reloaded = _loads(_dumps(original))
    _assert_array_equal(original, reloaded)


def test_jsonlinesio_numpy_custom_hooks():
    original = dict(array=np.array([100, 200, 300], dtype=np.int32), name="hundreds")

    reloaded = _loads(_dumps(original))
    _assert_array_equal(original["array"], reloaded["array"])
    assert reloaded["name"] == original["name"]

    reloaded_pairs = _loads(_dumps(original), load_args=dict(object_pairs_hook=tuple))
    assert reloaded_pairs[0][0] == "array"
    _assert_array_equal(reloaded_pairs[0][1], original["array"])
    assert reloaded_pairs[1] == ("name", "hundreds")


def test_jsonlinesio_failure():
    with pytest.raises(TypeError) as error:
        _dumps(CustomType())
    assert "CustomType" in str(error.value)


def test_jsonlinesio_context():
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

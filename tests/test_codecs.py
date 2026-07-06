import pytest

from dload.codecs import (
    json_bytes,
    json_from,
    npy_bytes,
    npy_from,
    text_bytes,
    text_from,
)


def test_json_round_trip_dict():
    obj = {"a": 1, "b": [1, 2, 3], "c": None, "d": True}
    assert json_from(json_bytes(obj)) == obj


def test_json_round_trip_unicode():
    obj = {"key": "日本語 émoji 🎉"}
    data = json_bytes(obj)
    assert json_from(data) == obj


def test_json_bytes_is_compact():
    data = json_bytes({"a": 1, "b": 2})
    assert b" " not in data


def test_json_bytes_returns_bytes():
    assert isinstance(json_bytes({"x": 1}), bytes)


def test_text_round_trip():
    s = "hello, world! 日本語"
    assert text_from(text_bytes(s)) == s


def test_text_bytes_is_utf8():
    assert text_bytes("é") == "é".encode("utf-8")


def test_npy_round_trip():
    np = pytest.importorskip("numpy")
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    data = npy_bytes(arr)
    assert isinstance(data, bytes)
    out = npy_from(data)
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == arr.dtype


def test_npy_round_trip_various_dtypes():
    np = pytest.importorskip("numpy")
    for dtype in (np.int16, np.int64, np.float64, np.bool_):
        arr = np.zeros((2, 3), dtype=dtype)
        arr.flat[0] = 1
        out = npy_from(npy_bytes(arr))
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == arr.dtype


def test_npy_from_disallows_pickle():
    np = pytest.importorskip("numpy")
    import io

    obj_arr = np.array([{"a": 1}], dtype=object)
    buf = io.BytesIO()
    np.save(buf, obj_arr, allow_pickle=True)
    with pytest.raises(ValueError):
        npy_from(buf.getvalue())

import pyarrow.feather as feather
import polars as pl

from flyhigh.data import download


def test_download_skips_files_already_present_with_right_size(tmp_path, monkeypatch):
    name = download.FILES[0].name
    pl.DataFrame({"a": [1]}).write_ipc(tmp_path / name)
    size = (tmp_path / name).stat().st_size
    monkeypatch.setattr(download, "_remote_size", lambda url: size)
    calls = []
    monkeypatch.setattr(download, "_stream", lambda url, dest, size: calls.append(url))
    download.download_all(tmp_path, only=[name])
    assert calls == []


def test_download_fetches_missing_file_and_verifies_it_opens(tmp_path, monkeypatch):
    name = download.FILES[0].name
    def fake_stream(url, dest, size):
        pl.DataFrame({"a": [1, 2]}).write_ipc(dest)
    monkeypatch.setattr(download, "_remote_size", lambda url: 0)
    monkeypatch.setattr(download, "_stream", fake_stream)
    download.download_all(tmp_path, only=[name])
    assert feather.read_table(tmp_path / name).num_rows == 2

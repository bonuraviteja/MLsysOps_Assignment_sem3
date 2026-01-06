from __future__ import annotations

from heart_disease_mlops.data.download import download_uci_dataset


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_download_uci_dataset_normalizes(tmp_path, monkeypatch) -> None:
    # Minimal two-row sample matching processed.cleveland.data format
    sample = (
        b"63.0,1.0,3.0,145.0,233.0,1.0,0.0,150.0,0.0,2.3,0.0,0.0,1.0,0\n"
        b"67.0,1.0,2.0,160.0,286.0,0.0,0.0,108.0,1.0,1.5,1.0,3.0,2.0,2\n"
    )

    import src.heart_disease_mlops.data.download as dl

    def fake_get(url: str, timeout: int):
        return _FakeResp(sample)

    monkeypatch.setattr(dl.requests, "get", fake_get)

    out_csv = tmp_path / "heart.csv"
    csv_path, sha = download_uci_dataset(output_csv=out_csv, force=True)
    assert csv_path.exists()
    assert isinstance(sha, str) and len(sha) == 64

    text = csv_path.read_text(encoding="utf-8")
    assert "target" in text
    # One row has num=0 => target 0; other num=2 => target 1
    assert ",0\n" in text
    assert ",1\n" in text

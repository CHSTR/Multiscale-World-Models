import pandas as pd

from probe_report import aggregate


def test_aggregate_keeps_latest_run_per_key(tmp_path):
    df = pd.DataFrame(
        [
            {
                "checkpoint": "a/weights.pt",
                "source": "cls",
                "probe": "ridge",
                "group": "g",
                "dim": 3,
                "mse_z": 0.1,
                "mse_raw": 0.1,
                "pearson": 0.9,
                "run": 0,
            },
            {
                "checkpoint": "a/weights.pt",
                "source": "cls",
                "probe": "ridge",
                "group": "g",
                "dim": 3,
                "mse_z": 0.05,
                "mse_raw": 0.05,
                "pearson": 0.95,
                "run": 1,
            },
        ]
    )
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    out = aggregate([str(p)])
    assert len(out) == 1
    assert out.iloc[0]["pearson"] == 0.95
    assert out.iloc[0]["model"] == "a"

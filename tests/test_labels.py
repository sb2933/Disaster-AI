from pathlib import Path

import pandas as pd
import pytest

from agents.seismic.labels import build_labels

# A single grid cell needs >=5 events anywhere in the catalog to be "active"
# (see features.select_active_cells), so 5 small filler events plus one
# M5.2 target event are placed at the same coordinates. A lone far-away
# event on 2020-12-31 just extends the catalog's date range so we can also
# exercise the "insufficient future data" NaN tail.
LAT, LON = 29.0, 84.0
RAW_COLUMNS = ["event_id", "time", "lat", "lon", "depth_km", "mag", "mag_type"]


def _row(event_id: str, date_str: str, mag: float, lat: float = LAT, lon: float = LON) -> dict:
    return {
        "event_id": event_id,
        "time": f"{date_str}T00:00:00+00:00",
        "lat": lat,
        "lon": lon,
        "depth_km": 10.0,
        "mag": mag,
        "mag_type": "mb",
    }


@pytest.fixture
def labels_df(tmp_path: Path) -> pd.DataFrame:
    rows = [_row(f"filler{i}", f"2020-01-0{i}", 3.5) for i in range(1, 6)]
    rows.append(_row("mainshock", "2020-06-15", 5.2))
    rows.append(_row("far_filler", "2020-12-31", 3.0, lat=26.0, lon=75.0))
    catalog = pd.DataFrame(rows, columns=RAW_COLUMNS)

    raw_path = tmp_path / "catalog.csv"
    out_path = tmp_path / "labels.parquet"
    catalog.to_csv(raw_path, index=False)
    return build_labels(raw_path, out_path)


def _cell_labels(labels_df: pd.DataFrame) -> pd.Series:
    # Find the target cell (the one containing the mainshock) by looking for
    # whichever cell has positive labels at all.
    candidate_ids = labels_df.loc[labels_df["label"] == 1, "cell_id"].unique()
    assert len(candidate_ids) == 1, "expected exactly one active cell with a positive label"
    cell_id = candidate_ids[0]
    return labels_df[labels_df["cell_id"] == cell_id].set_index("date")["label"]


def test_label_is_one_exactly_seven_days_before_event(labels_df: pd.DataFrame):
    series = _cell_labels(labels_df)
    positive_dates = series[series == 1.0].index
    expected = pd.date_range("2020-06-08", "2020-06-14", freq="D")
    assert set(positive_dates) == set(expected)


def test_label_is_zero_just_outside_the_window(labels_df: pd.DataFrame):
    series = _cell_labels(labels_df)
    assert series.loc[pd.Timestamp("2020-06-07")] == 0.0
    assert series.loc[pd.Timestamp("2020-06-15")] == 0.0


def test_label_is_nan_when_future_data_insufficient(labels_df: pd.DataFrame):
    series = _cell_labels(labels_df)
    # Catalog's last date is 2020-12-31; t+7 > 2020-12-31 for t >= 2020-12-25.
    assert series.loc[pd.Timestamp("2020-12-24")] == 0.0
    assert series.loc[pd.Timestamp("2020-12-25") :].isna().all()


def test_no_nan_labels_well_before_catalog_end(labels_df: pd.DataFrame):
    series = _cell_labels(labels_df)
    assert series.loc[:pd.Timestamp("2020-12-24")].notna().all()

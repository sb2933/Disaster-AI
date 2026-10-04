"""Fetch earthquake catalog data from the USGS FDSN event API.

USGS caps any single query at 20,000 events. We stay well clear of that by
querying year-by-year, and further splitting any year that still hits the
cap (via binary split on the time range). Network calls are retried with
exponential backoff and a short delay is inserted between requests to be a
polite API citizen.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_EVENT_CAP = 20000
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF_S = 5
INTER_REQUEST_DELAY_S = 1.0

RAW_CSV_COLUMNS = ["event_id", "time", "lat", "lon", "depth_km", "mag", "mag_type"]

DEFAULT_RAW_PATH = Path("data/raw/usgs_catalog.csv")

# Training region bounding box: Himalayan belt, lat 25-36N, lon 70-97E.
DEFAULT_BBOX = {"min_lat": 25.0, "max_lat": 36.0, "min_lon": 70.0, "max_lon": 97.0}


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def _request_with_retries(params: dict) -> requests.Response:
    """GET the USGS API, retrying transient failures with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(USGS_URL, params=params, timeout=REQUEST_TIMEOUT_S)
            if response.status_code == 200:
                return response
            if response.status_code == 204:
                # No content: valid "zero events in this window" response.
                return response
            logger.warning(
                "USGS request failed (status=%s), attempt %d/%d: %s",
                response.status_code, attempt, MAX_RETRIES, response.text[:200],
            )
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("USGS request error, attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
        time.sleep(RETRY_BACKOFF_S * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"USGS request failed after {MAX_RETRIES} attempts")


def _count_events(start: datetime, end: datetime, bbox: BBox, min_mag: float) -> int:
    """Query the API's `count` endpoint to check whether a window exceeds the event cap."""
    params = {
        "format": "text",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "minlatitude": bbox.min_lat,
        "maxlatitude": bbox.max_lat,
        "minlongitude": bbox.min_lon,
        "maxlongitude": bbox.max_lon,
    }
    count_params = dict(params)
    response = requests.get(
        USGS_URL.replace("/query", "/count"), params=count_params, timeout=REQUEST_TIMEOUT_S
    )
    response.raise_for_status()
    text = response.text.strip()
    return int(text) if text else 0


def _parse_geojson(payload: dict) -> pd.DataFrame:
    rows = []
    for feature in payload.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"] if feature.get("geometry") else [None, None, None]
        lon, lat, depth_km = (coords + [None, None, None])[:3]
        rows.append(
            {
                "event_id": feature.get("id"),
                "time": pd.to_datetime(props.get("time"), unit="ms", utc=True),
                "lat": lat,
                "lon": lon,
                "depth_km": depth_km,
                "mag": props.get("mag"),
                "mag_type": props.get("magType"),
            }
        )
    return pd.DataFrame(rows, columns=RAW_CSV_COLUMNS)


def _fetch_window(start: datetime, end: datetime, bbox: BBox, min_mag: float) -> pd.DataFrame:
    """Fetch one time window, splitting recursively if it would exceed USGS's event cap."""
    count = _count_events(start, end, bbox, min_mag)
    if count > USGS_EVENT_CAP:
        if end - start <= timedelta(days=1):
            logger.warning(
                "Window %s..%s has %d events but cannot be split further; truncating to cap.",
                start, end, count,
            )
        else:
            mid = start + (end - start) / 2
            logger.info("Window %s..%s has %d events (> cap); splitting at %s", start, end, count, mid)
            left = _fetch_window(start, mid, bbox, min_mag)
            right = _fetch_window(mid, end, bbox, min_mag)
            return pd.concat([left, right], ignore_index=True)

    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "minlatitude": bbox.min_lat,
        "maxlatitude": bbox.max_lat,
        "minlongitude": bbox.min_lon,
        "maxlongitude": bbox.max_lon,
        "limit": USGS_EVENT_CAP,
        "orderby": "time-asc",
    }
    logger.info("Fetching %s..%s (%d events expected)", start, end, count)
    response = _request_with_retries(params)
    time.sleep(INTER_REQUEST_DELAY_S)
    if response.status_code == 204 or not response.text.strip():
        return pd.DataFrame(columns=RAW_CSV_COLUMNS)
    payload = response.json()
    return _parse_geojson(payload)


def fetch_historical(
    start: date,
    end: date,
    bbox: BBox = BBox(**DEFAULT_BBOX),
    min_mag: float = 3.0,
    out_path: Path = DEFAULT_RAW_PATH,
) -> pd.DataFrame:
    """Fetch the full historical catalog for [start, end) by querying year-by-year.

    Each year is further split in half (recursively) if it would exceed the
    USGS 20,000-event response cap. Results are written to ``out_path`` as
    CSV (overwriting any existing file) and also returned as a DataFrame.
    """
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)

    frames: list[pd.DataFrame] = []
    year_start = start_dt
    while year_start < end_dt:
        year_end = min(datetime(year_start.year + 1, 1, 1, tzinfo=timezone.utc), end_dt)
        frames.append(_fetch_window(year_start, year_end, bbox, min_mag))
        year_start = year_end

    catalog = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_CSV_COLUMNS)
    catalog = catalog.drop_duplicates(subset="event_id").sort_values("time").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)
    logger.info("Wrote %d events to %s", len(catalog), out_path)
    return catalog


def fetch_recent(
    days: int = 2,
    bbox: BBox = BBox(**DEFAULT_BBOX),
    min_mag: float = 3.0,
    out_path: Path = DEFAULT_RAW_PATH,
) -> pd.DataFrame:
    """Fetch the last ``days`` days of events and merge into the existing catalog file.

    Dedupes on event_id so this is safe to run repeatedly (e.g. daily cron).
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    new_events = _fetch_window(start_dt, end_dt, bbox, min_mag)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing["time"] = pd.to_datetime(existing["time"], utc=True, format="ISO8601")
        combined = pd.concat([existing, new_events], ignore_index=True) if not new_events.empty else existing
    else:
        combined = new_events

    combined = combined.drop_duplicates(subset="event_id").sort_values("time").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Fetched %d new/updated events; catalog now has %d rows -> %s", len(new_events), len(combined), out_path)
    return combined


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch USGS earthquake catalog data.")
    sub = parser.add_subparsers(dest="command", required=True)

    hist = sub.add_parser("historical", help="Fetch the full historical catalog.")
    hist.add_argument("--start", type=_parse_date, default=date(1975, 1, 1))
    hist.add_argument("--end", type=_parse_date, default=date.today())
    hist.add_argument("--min-mag", type=float, default=3.0)
    hist.add_argument("--min-lat", type=float, default=DEFAULT_BBOX["min_lat"])
    hist.add_argument("--max-lat", type=float, default=DEFAULT_BBOX["max_lat"])
    hist.add_argument("--min-lon", type=float, default=DEFAULT_BBOX["min_lon"])
    hist.add_argument("--max-lon", type=float, default=DEFAULT_BBOX["max_lon"])
    hist.add_argument("--out", type=Path, default=DEFAULT_RAW_PATH)

    recent = sub.add_parser("recent", help="Fetch recent events and merge into the catalog.")
    recent.add_argument("--days", type=int, default=2)
    recent.add_argument("--min-mag", type=float, default=3.0)
    recent.add_argument("--min-lat", type=float, default=DEFAULT_BBOX["min_lat"])
    recent.add_argument("--max-lat", type=float, default=DEFAULT_BBOX["max_lat"])
    recent.add_argument("--min-lon", type=float, default=DEFAULT_BBOX["min_lon"])
    recent.add_argument("--max-lon", type=float, default=DEFAULT_BBOX["max_lon"])
    recent.add_argument("--out", type=Path, default=DEFAULT_RAW_PATH)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "historical":
        bbox = BBox(args.min_lat, args.max_lat, args.min_lon, args.max_lon)
        fetch_historical(args.start, args.end, bbox, args.min_mag, args.out)
    elif args.command == "recent":
        bbox = BBox(args.min_lat, args.max_lat, args.min_lon, args.max_lon)
        fetch_recent(args.days, bbox, args.min_mag, args.out)


if __name__ == "__main__":
    main()

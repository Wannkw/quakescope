"""Download and cache the USGS ComCat earthquake catalog."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"]


def fetch_year(year: int, min_mag: float = 4.5, retries: int = 3) -> pd.DataFrame:
    """Fetch one calendar year of events (USGS caps a query at 20,000 rows)."""
    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year + 1}-01-01",
        "minmagnitude": min_mag,
        "eventtype": "earthquake",
        "orderby": "time-asc",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(USGS_URL, params=params, timeout=120)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), usecols=COLUMNS)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_catalog(
    start_year: int = 2000,
    end_year: int = 2025,
    min_mag: float = 4.5,
    cache_dir: Path | str = "data",
) -> pd.DataFrame:
    """Return a cleaned catalog, downloading any year not already cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start_year, end_year + 1):
        path = cache_dir / f"usgs_{year}_m{min_mag}.csv"
        if not path.exists():
            print(f"  downloading {year} ...")
            fetch_year(year, min_mag).to_csv(path, index=False)
        frames.append(pd.read_csv(path))
    return clean(pd.concat(frames, ignore_index=True), min_mag)


def clean(df: pd.DataFrame, min_mag: float = 4.5) -> pd.DataFrame:
    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
    # The API occasionally returns events below minmagnitude (revised magnitudes).
    df = df[df["mag"] >= min_mag].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.drop_duplicates("id").sort_values("time").reset_index(drop=True)
    # Days since the first event: a float axis that is convenient for window math.
    df["t_days"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 86400.0
    return df

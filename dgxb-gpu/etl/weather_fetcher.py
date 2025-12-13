"""
Open-Meteo Weather Data Fetcher for DGXB (GPU)
Pulls weather data from Open-Meteo API (https://open-meteo.com/)
No API key required.

Notes:
- Network I/O (requests) is CPU-bound.
- Response parsing + dataframe ops are GPU-accelerated with cuDF.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _require_gpu_libs():
    try:
        import cudf  # noqa: F401
        import pyarrow as pa  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GPU libraries not available. Install RAPIDS cuDF + pyarrow "
            "matching your CUDA runtime. "
            f"Original error: {e}"
        )


@dataclass(frozen=True)
class OpenMeteoConfig:
    rate_limit_delay: float = 0.1  # permissive; keep to be a good citizen
    max_workers: int = 8  # for parallel location fetch
    use_threads: bool = True  # parallelize requests across locations


class OpenMeteoWeatherFetcherGPU:
    """
    Fetches weather data from Open-Meteo API and returns cuDF DataFrames.
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # DGXB canonical hourly feature list
    HOURLY_FIELDS = [
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "precipitation_probability",
        "weather_code",
    ]

    def __init__(self, cfg: Optional[OpenMeteoConfig] = None):
        self.cfg = cfg or OpenMeteoConfig()
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "DGXB/1.0 (transportation-intelligence@example.com)"}
        )

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.cfg.rate_limit_delay:
            time.sleep(self.cfg.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url}, error: {e}")
            raise

    @staticmethod
    def _normalize_hourly_payload_to_cudf(
        data: Dict[str, Any],
        lat: float,
        lon: float,
        source: str,
        start_dt: datetime,
        end_dt: datetime,
    ):
        """
        Convert Open-Meteo response JSON into a cuDF DataFrame WITHOUT per-row loops.
        Filters timestamps to [start_dt, end_dt] inclusive.
        """
        _require_gpu_libs()
        import cudf
        import pyarrow as pa

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return cudf.DataFrame()

        # Build Arrow arrays first for reliable typing (and faster ingest)
        # Timestamps: Open-Meteo returns ISO strings, often without 'Z' if timezone=auto
        # We'll parse on GPU to UTC.
        cols: Dict[str, Any] = {"timestamp": times}

        # Pull arrays (must be same length as times)
        def safe_get(name: str):
            arr = hourly.get(name)
            if arr is None:
                return [None] * len(times)
            # Sometimes API returns shorter arrays; guard defensively
            if len(arr) != len(times):
                if len(arr) < len(times):
                    return arr + [None] * (len(times) - len(arr))
                return arr[: len(times)]
            return arr

        cols["temperature"] = safe_get("temperature_2m")
        cols["humidity"] = safe_get("relative_humidity_2m")
        cols["dewpoint"] = safe_get("dewpoint_2m")
        cols["wind_speed"] = safe_get("wind_speed_10m")
        cols["wind_direction"] = safe_get("wind_direction_10m")
        cols["precipitation_amount"] = safe_get("precipitation")
        cols["precipitation_probability"] = safe_get("precipitation_probability")
        cols["weather_code"] = safe_get("weather_code")

        # Constant columns
        cols["lat"] = [lat] * len(times)
        cols["lon"] = [lon] * len(times)
        cols["data_source"] = [source] * len(times)

        table = pa.Table.from_pydict(cols)
        gdf = cudf.DataFrame.from_arrow(table)

        # Parse timestamp to UTC on GPU
        # If Open-Meteo returns local time due to timezone=auto, this will interpret as naive.
        # We force UTC by requesting timezone=UTC in params (recommended below).
        try:
            gdf["timestamp"] = cudf.to_datetime(
                gdf["timestamp"], utc=True, errors="coerce"
            )
        except Exception:
            gdf["timestamp"] = cudf.to_datetime(gdf["timestamp"], utc=True)

        # Numeric coercions
        for c in [
            "temperature",
            "humidity",
            "dewpoint",
            "wind_speed",
            "wind_direction",
            "precipitation_amount",
            "precipitation_probability",
            "weather_code",
            "lat",
            "lon",
        ]:
            if c in gdf.columns:
                try:
                    gdf[c] = cudf.to_numeric(gdf[c], errors="coerce")
                except Exception:
                    gdf[c] = cudf.to_numeric(gdf[c].astype("str"), errors="coerce")

        # Filter to requested window (convert python datetimes to pandas timestamps for comparison)
        start_ts = pd.Timestamp(start_dt.astimezone(timezone.utc))
        end_ts = pd.Timestamp(end_dt.astimezone(timezone.utc))
        gdf = gdf[(gdf["timestamp"] >= start_ts) & (gdf["timestamp"] <= end_ts)]

        return gdf

    def fetch_historical_weather_gpu(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ):
        """
        Fetch historical weather data for a location and time range (cuDF).
        """
        _require_gpu_libs()
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(self.HOURLY_FIELDS),
            # IMPORTANT: use UTC so timestamps are unambiguous and consistent with traffic
            "timezone": "UTC",
        }

        try:
            data = self._make_request(self.ARCHIVE_URL, params)
        except Exception as e:
            logger.warning(f"Failed historical weather for ({lat}, {lon}): {e}")
            import cudf

            return cudf.DataFrame()

        return self._normalize_hourly_payload_to_cudf(
            data=data,
            lat=lat,
            lon=lon,
            source="historical",
            start_dt=start_date,
            end_dt=end_date,
        )

    def fetch_forecast_weather_gpu(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ):
        """
        Fetch forecast weather data for a location and time range (cuDF).
        """
        _require_gpu_libs()
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(self.HOURLY_FIELDS),
            "timezone": "UTC",
        }

        try:
            data = self._make_request(self.FORECAST_URL, params)
        except Exception as e:
            logger.warning(f"Failed forecast weather for ({lat}, {lon}): {e}")
            import cudf

            return cudf.DataFrame()

        return self._normalize_hourly_payload_to_cudf(
            data=data,
            lat=lat,
            lon=lon,
            source="forecast",
            start_dt=start_date,
            end_dt=end_date,
        )

    def fetch_weather_for_locations_gpu(
        self,
        locations: List[Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = True,
        include_forecast: bool = True,
    ):
        """
        Fetch both historical and forecast weather for multiple locations.
        Returns one combined cuDF DataFrame.
        """
        _require_gpu_libs()
        import cudf

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        def fetch_one(lat: float, lon: float):
            parts = []
            # Historical (<= now)
            if include_historical:
                hist_end = min(end_date, now)
                if start_date < hist_end:
                    gdf_h = self.fetch_historical_weather_gpu(
                        lat, lon, start_date, hist_end
                    )
                    if gdf_h is not None and len(gdf_h) > 0:
                        parts.append(gdf_h)

            # Forecast (>= now)
            if include_forecast:
                fc_start = max(start_date, now)
                if fc_start <= end_date:
                    gdf_f = self.fetch_forecast_weather_gpu(
                        lat, lon, fc_start, end_date
                    )
                    if gdf_f is not None and len(gdf_f) > 0:
                        parts.append(gdf_f)

            if not parts:
                return None
            return cudf.concat(parts, ignore_index=True)

        all_parts = []

        if self.cfg.use_threads and len(locations) > 1:
            # Parallelize across locations (network-limited)
            max_workers = max(1, int(self.cfg.max_workers))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(fetch_one, lat, lon): (lat, lon) for lat, lon in locations
                }
                for fut in as_completed(futs):
                    lat, lon = futs[fut]
                    try:
                        gdf = fut.result()
                        if gdf is not None and len(gdf) > 0:
                            all_parts.append(gdf)
                            logger.info(
                                f"Fetched {len(gdf)} weather rows for ({lat}, {lon})"
                            )
                    except Exception as e:
                        logger.error(f"Failed weather for ({lat}, {lon}): {e}")
        else:
            for lat, lon in locations:
                try:
                    gdf = fetch_one(lat, lon)
                    if gdf is not None and len(gdf) > 0:
                        all_parts.append(gdf)
                        logger.info(
                            f"Fetched {len(gdf)} weather rows for ({lat}, {lon})"
                        )
                except Exception as e:
                    logger.error(f"Failed weather for ({lat}, {lon}): {e}")

        if not all_parts:
            return cudf.DataFrame()

        combined = cudf.concat(all_parts, ignore_index=True)

        # Sort on GPU
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)

        # Optional: de-dupe exact duplicates (API sometimes overlaps)
        combined = combined.drop_duplicates()

        return combined


def save_weather_bronze_gpu(
    gdf,
    output_dir: str = "bronze-weather",
    partition_by_date: bool = True,
):
    """
    Save weather data to bronze layer (GPU write).
    Partitions by date via date_str and writes one parquet per date.
    """
    _require_gpu_libs()
    import cudf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if gdf is None or len(gdf) == 0:
        logger.warning("Empty weather DataFrame, nothing to save")
        return

    if partition_by_date and "timestamp" in gdf.columns:
        gdf = gdf.copy()
        # date_str = YYYY-MM-DD (GPU)
        try:
            gdf["date_str"] = gdf["timestamp"].dt.strftime("%Y-%m-%d")
        except Exception:
            # fallback: do only this on CPU
            pdf = gdf[["timestamp"]].to_pandas()
            pdf["date_str"] = pd.to_datetime(pdf["timestamp"], utc=True).dt.strftime(
                "%Y-%m-%d"
            )
            gdf["date_str"] = cudf.from_pandas(pdf["date_str"])

        for date_str, group in gdf.groupby("date_str"):
            file_path = output_path / f"weather_{date_str}.parquet"
            group = group.drop(columns=["date_str"])
            group.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(group)} records to {file_path}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"weather_{ts}.parquet"
        gdf.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(gdf)} records to {file_path}")

    # Metadata (minimal CPU extraction)
    date_start = None
    date_end = None
    locations = []
    sources = []

    try:
        if "timestamp" in gdf.columns:
            tmin = gdf["timestamp"].min()
            tmax = gdf["timestamp"].max()
            if tmin is not None and tmax is not None:
                date_start = pd.Timestamp(tmin.to_pandas()).isoformat()
                date_end = pd.Timestamp(tmax.to_pandas()).isoformat()
        if "lat" in gdf.columns and "lon" in gdf.columns:
            # pull unique locations (small)
            loc_pdf = gdf[["lat", "lon"]].dropna().drop_duplicates().to_pandas()
            locations = loc_pdf.to_dict("records")
        if "data_source" in gdf.columns:
            sources = gdf["data_source"].dropna().unique().to_pandas().tolist()
    except Exception:
        pass

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": int(len(gdf)),
        "columns": list(gdf.columns),
        "date_range": {"start": date_start, "end": date_end},
        "locations": locations,
        "data_sources": sources,
        "engine": "cudf",
        "provider": "open-meteo",
        "timezone": "UTC",
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def fetch_and_save_weather_gpu(
    locations: List[Tuple[float, float]],
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "bronze-weather",
    include_historical: bool = True,
    include_forecast: bool = True,
    cfg: Optional[OpenMeteoConfig] = None,
):
    """
    Convenience function: Fetch weather data and save to bronze-weather (GPU write).
    """
    fetcher = OpenMeteoWeatherFetcherGPU(cfg=cfg)

    logger.info(
        f"Fetching weather for {len(locations)} locations from {start_date} to {end_date} "
        f"(historical={include_historical}, forecast={include_forecast})"
    )

    gdf = fetcher.fetch_weather_for_locations_gpu(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        include_historical=include_historical,
        include_forecast=include_forecast,
    )

    if gdf is None or len(gdf) == 0:
        logger.warning("No weather data fetched")
        return None

    save_weather_bronze_gpu(gdf, output_dir=output_dir, partition_by_date=True)
    logger.info(f"Weather data saved to {output_dir}")

    return gdf


if __name__ == "__main__":
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # Austin, TX locations
    locations = [
        (30.2672, -97.7431),  # Downtown Austin
        (30.3072, -97.7559),  # North Austin
    ]

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 7, tzinfo=timezone.utc)

    fetch_and_save_weather_gpu(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        output_dir="bronze-weather",
        include_historical=True,
        include_forecast=True,
        cfg=OpenMeteoConfig(rate_limit_delay=0.1, max_workers=8, use_threads=True),
    )

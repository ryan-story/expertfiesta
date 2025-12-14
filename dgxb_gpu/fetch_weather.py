#!/usr/bin/env python3
"""
Fetch weather data and save to bronze layer.

Usage examples:
  python fetch_weather.py --days 30 --output-dir bronze-weather
  python fetch_weather.py --start 2025-11-01 --end 2025-12-01 --historical-only
  python fetch_weather.py --locations 30.2672,-97.7431 30.3072,-97.7559
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import argparse

# Add parent directory to path (repo-local import)
sys.path.insert(0, str(Path(__file__).parent))

from etl.weather_fetcher import fetch_and_save_weather  # noqa: E402


logger = logging.getLogger("fetch_weather")


def parse_locations(vals: List[str]) -> List[Tuple[float, float]]:
    locs: List[Tuple[float, float]] = []
    for v in vals:
        try:
            lat_s, lon_s = v.split(",", 1)
            lat = float(lat_s.strip())
            lon = float(lon_s.strip())
        except Exception as e:
            raise ValueError(f"Invalid location '{v}'. Expected 'lat,lon'. Error: {e}")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Location out of bounds: {v}")
        locs.append((lat, lon))
    return locs


def parse_date(s: str) -> datetime:
    # Accept YYYY-MM-DD; interpret as UTC midnight
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locations",
        nargs="*",
        default=[
            "30.2672,-97.7431",  # Downtown Austin
            "30.3072,-97.7559",  # North Austin
            "30.2244,-97.7694",  # South Austin
        ],
        help="Space-separated list of 'lat,lon' pairs",
    )
    parser.add_argument(
        "--output-dir", default="bronze-weather", help="Output directory"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback days (ignored if --start/--end provided)",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start date UTC (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date UTC (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--historical-only", action="store_true", help="Fetch historical only"
    )
    parser.add_argument(
        "--forecast-only", action="store_true", help="Fetch forecast only"
    )
    parser.add_argument(
        "--max-days", type=int, default=365, help="Safety cap for requested range"
    )
    args = parser.parse_args(argv)

    if args.historical_only and args.forecast_only:
        logger.error("Cannot set both --historical-only and --forecast-only")
        return 2

    locations = parse_locations(args.locations)

    # Determine date window (UTC-aware)
    if args.start or args.end:
        if not (args.start and args.end):
            logger.error("If using --start/--end, you must provide both")
            return 2
        start_date = parse_date(args.start)
        end_date = (
            parse_date(args.end) + timedelta(days=1) - timedelta(seconds=1)
        )  # inclusive end day
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)

    # Safety cap
    span_days = (end_date - start_date).total_seconds() / 86400.0
    if span_days > args.max_days:
        logger.error(
            f"Requested range is {span_days:.1f} days, exceeds --max-days={args.max_days}"
        )
        return 2

    include_historical = not args.forecast_only
    include_forecast = not args.historical_only

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching weather for {len(locations)} locations")
    logger.info(f"Date range (UTC): {start_date.isoformat()} to {end_date.isoformat()}")
    logger.info(
        f"include_historical={include_historical}, include_forecast={include_forecast}"
    )
    logger.info(f"Output dir: {out_dir.resolve()}")

    weather_df = fetch_and_save_weather(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(out_dir),
        include_historical=include_historical,
        include_forecast=include_forecast,
    )

    if weather_df is not None and not weather_df.empty:
        # Ensure timestamp is printed clearly
        ts_min = weather_df["timestamp"].min()
        ts_max = weather_df["timestamp"].max()
        logger.info(f"Success: fetched {len(weather_df):,} records")
        logger.info(f"Columns: {list(weather_df.columns)}")
        logger.info(f"Timestamp range: {ts_min} to {ts_max}")
        return 0

    logger.warning("No weather data fetched (empty result). Check upstream logs.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())

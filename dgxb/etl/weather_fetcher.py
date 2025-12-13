"""
Open-Meteo Weather Data Fetcher for DGXB
Pulls weather data from Open-Meteo API (https://open-meteo.com/)
No API key required, free and open source.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

logger = logging.getLogger(__name__)


class OpenMeteoWeatherFetcher:
    """
    Fetches weather data from Open-Meteo API
    No API key required
    Rate limit: 10,000 requests/day (very generous)
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, rate_limit_delay: float = 0.1):  # Very permissive rate limit
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: dict) -> dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url}, error: {e}")
            raise

    def fetch_historical_weather(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for a location and time range
        Returns DataFrame compatible with DGXB data product schema
        """
        # Open-Meteo expects dates in YYYY-MM-DD format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,"
                     "wind_speed_10m,wind_direction_10m,precipitation,"
                     "precipitation_probability,weather_code",
            "timezone": "auto",  # Auto-detect timezone
        }

        try:
            data = self._make_request(self.ARCHIVE_URL, params)
        except Exception as e:
            logger.warning(f"Failed to fetch historical weather for ({lat}, {lon}): {e}")
            return pd.DataFrame()

        # Parse Open-Meteo response
        hourly = data.get("hourly", {})
        if not hourly:
            logger.warning(f"No hourly data in response for ({lat}, {lon})")
            return pd.DataFrame()

        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        # Build DataFrame
        weather_data = []
        for i, time_str in enumerate(times):
            try:
                # Parse timestamp (Open-Meteo returns ISO format)
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                
                # Only include data within requested range
                if timestamp < start_date or timestamp > end_date:
                    continue

                weather_data.append(
                    {
                        "timestamp": timestamp,
                        "lat": lat,
                        "lon": lon,
                        "temperature": hourly.get("temperature_2m", [None])[i],
                        "dewpoint": hourly.get("dewpoint_2m", [None])[i],
                        "humidity": hourly.get("relative_humidity_2m", [None])[i],
                        "wind_speed": hourly.get("wind_speed_10m", [None])[i],
                        "wind_direction": hourly.get("wind_direction_10m", [None])[i],
                        "precipitation_amount": hourly.get("precipitation", [None])[i],
                        "precipitation_probability": hourly.get("precipitation_probability", [None])[i],
                        "weather_code": hourly.get("weather_code", [None])[i],
                        "data_source": "historical",  # Mark as historical
                    }
                )
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"Error parsing weather data at index {i}: {e}")
                continue

        if not weather_data:
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        return df

    def fetch_forecast_weather(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch forecast weather data for a location and time range
        Returns DataFrame compatible with DGXB data product schema
        """
        # Open-Meteo expects dates in YYYY-MM-DD format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,"
                     "wind_speed_10m,wind_direction_10m,precipitation,"
                     "precipitation_probability,weather_code",
            "timezone": "auto",  # Auto-detect timezone
        }

        try:
            data = self._make_request(self.FORECAST_URL, params)
        except Exception as e:
            logger.warning(f"Failed to fetch forecast weather for ({lat}, {lon}): {e}")
            return pd.DataFrame()

        # Parse Open-Meteo response
        hourly = data.get("hourly", {})
        if not hourly:
            logger.warning(f"No hourly data in forecast response for ({lat}, {lon})")
            return pd.DataFrame()

        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        # Build DataFrame
        weather_data = []
        for i, time_str in enumerate(times):
            try:
                # Parse timestamp (Open-Meteo returns ISO format)
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                
                # Only include data within requested range
                if timestamp < start_date or timestamp > end_date:
                    continue

                weather_data.append(
                    {
                        "timestamp": timestamp,
                        "lat": lat,
                        "lon": lon,
                        "temperature": hourly.get("temperature_2m", [None])[i],
                        "dewpoint": hourly.get("dewpoint_2m", [None])[i],
                        "humidity": hourly.get("relative_humidity_2m", [None])[i],
                        "wind_speed": hourly.get("wind_speed_10m", [None])[i],
                        "wind_direction": hourly.get("wind_direction_10m", [None])[i],
                        "precipitation_amount": hourly.get("precipitation", [None])[i],
                        "precipitation_probability": hourly.get("precipitation_probability", [None])[i],
                        "weather_code": hourly.get("weather_code", [None])[i],
                        "data_source": "forecast",  # Mark as forecast
                    }
                )
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"Error parsing forecast data at index {i}: {e}")
                continue

        if not weather_data:
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        return df

    def fetch_weather_for_locations(
        self,
        locations: List[Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = True,
        include_forecast: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch both historical and forecast weather data for multiple locations
        Returns combined DataFrame with data_source column indicating source
        
        Args:
            locations: List of (lat, lon) tuples
            start_date: Start date for data range
            end_date: End date for data range
            include_historical: Whether to fetch historical data
            include_forecast: Whether to fetch forecast data
        """
        all_weather = []
        now = datetime.now(start_date.tzinfo if start_date.tzinfo else None)
        if start_date.tzinfo is None:
            now = datetime.now()

        for lat, lon in locations:
            try:
                # Fetch historical data (up to today)
                if include_historical:
                    historical_end = min(end_date, now)
                    if start_date < historical_end:
                        df_historical = self.fetch_historical_weather(
                            lat, lon, start_date, historical_end
                        )
                        if not df_historical.empty:
                            all_weather.append(df_historical)
                            logger.info(
                                f"Fetched {len(df_historical)} historical records for ({lat}, {lon})"
                            )

                # Fetch forecast data (from today onwards)
                if include_forecast:
                    forecast_start = max(start_date, now)
                    if forecast_start <= end_date:
                        df_forecast = self.fetch_forecast_weather(
                            lat, lon, forecast_start, end_date
                        )
                        if not df_forecast.empty:
                            all_weather.append(df_forecast)
                            logger.info(
                                f"Fetched {len(df_forecast)} forecast records for ({lat}, {lon})"
                            )

            except Exception as e:
                logger.error(f"Failed to fetch weather for ({lat}, {lon}): {e}")
                continue

        if not all_weather:
            return pd.DataFrame()

        combined = pd.concat(all_weather, ignore_index=True)
        # Sort by timestamp
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined


def save_weather_bronze(
    weather_df: pd.DataFrame,
    output_dir: str = "bronze-weather",
    partition_by_date: bool = True,
):
    """
    Save weather data to bronze layer (raw ingested data)
    Follows data lakehouse bronze/silver/gold pattern
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if weather_df.empty:
        logger.warning("Empty weather DataFrame, nothing to save")
        return

    if partition_by_date and "timestamp" in weather_df.columns:
        # Partition by date for efficient querying
        weather_df = weather_df.copy()
        weather_df["date"] = pd.to_datetime(weather_df["timestamp"]).dt.date

        for date, group_df in weather_df.groupby("date"):
            date_str = date.strftime("%Y-%m-%d")
            file_path = output_path / f"weather_{date_str}.parquet"
            group_df.drop(columns=["date"]).to_parquet(file_path, index=False)
            logger.info(f"Saved {len(group_df)} records to {file_path}")
    else:
        # Single file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"weather_{timestamp_str}.parquet"
        weather_df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(weather_df)} records to {file_path}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(weather_df),
        "columns": list(weather_df.columns),
        "date_range": {
            "start": (
                weather_df["timestamp"].min().isoformat()
                if "timestamp" in weather_df.columns
                else None
            ),
            "end": (
                weather_df["timestamp"].max().isoformat()
                if "timestamp" in weather_df.columns
                else None
            ),
        },
        "locations": (
            weather_df[["lat", "lon"]].drop_duplicates().to_dict("records")
            if "lat" in weather_df.columns
            else []
        ),
        "data_sources": (
            weather_df["data_source"].unique().tolist()
            if "data_source" in weather_df.columns
            else []
        ),
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def fetch_and_save_weather(
    locations: List[Tuple[float, float]],
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "bronze-weather",
    include_historical: bool = True,
    include_forecast: bool = True,
):
    """
    Convenience function: Fetch weather data (historical + forecast) and save to bronze-weather layer
    """
    fetcher = OpenMeteoWeatherFetcher()

    logger.info(
        f"Fetching weather data for {len(locations)} locations from {start_date} to {end_date}"
    )
    logger.info(f"Include historical: {include_historical}, Include forecast: {include_forecast}")
    
    weather_df = fetcher.fetch_weather_for_locations(
        locations,
        start_date,
        end_date,
        include_historical=include_historical,
        include_forecast=include_forecast,
    )

    if weather_df.empty:
        logger.warning("No weather data fetched")
        return None

    save_weather_bronze(weather_df, output_dir)
    logger.info(f"Weather data saved to {output_dir}")
    
    # Print summary
    if "data_source" in weather_df.columns:
        source_counts = weather_df["data_source"].value_counts()
        logger.info(f"Data source breakdown: {source_counts.to_dict()}")

    return weather_df


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Austin, TX locations
    locations = [
        (30.2672, -97.7431),  # Downtown Austin
        (30.3072, -97.7559),  # North Austin
    ]

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)

    fetch_and_save_weather(locations, start_date, end_date)

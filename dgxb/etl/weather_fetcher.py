"""
NWS Weather Data Fetcher for DGXB
Pulls weather data from National Weather Service API (https://api.weather.gov/)
No API key required, but User-Agent header is required.
"""

import requests
import pandas as pd
from datetime import datetime
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

logger = logging.getLogger(__name__)


class NWSWeatherFetcher:
    """
    Fetches weather data from National Weather Service API
    No API key required, but User-Agent header is required
    Rate limit: 30 requests/minute
    """

    BASE_URL = "https://api.weather.gov"
    HEADERS = {
        "User-Agent": "DGXB/1.0 (transportation-intelligence@example.com)",
        "Accept": "application/json",
    }

    def __init__(self, rate_limit_delay: float = 2.1):  # 30 req/min = 2 sec/req
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        try:
            response = requests.get(
                url, headers=self.HEADERS, params=params, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url}, error: {e}")
            raise

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """Get grid point info for a location"""
        url = f"{self.BASE_URL}/points/{lat},{lon}"
        return self._make_request(url)

    def get_forecast(self, lat: float, lon: float, hourly: bool = False) -> dict:
        """Get forecast for a location"""
        grid = self.get_grid_point(lat, lon)
        office = grid["properties"]["gridId"]
        gridX = grid["properties"]["gridX"]
        gridY = grid["properties"]["gridY"]

        endpoint = "forecast/hourly" if hourly else "forecast"
        url = f"{self.BASE_URL}/gridpoints/{office}/{gridX},{gridY}/{endpoint}"
        return self._make_request(url)

    def get_observations(
        self,
        station_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict:
        """Get observations from a weather station"""
        url = f"{self.BASE_URL}/stations/{station_id}/observations"

        params = {}
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()

        return self._make_request(url, params)

    def fetch_forecast_for_location(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch forecast data for a location and time range
        Returns DataFrame compatible with DGXB data product schema
        """
        try:
            forecast = self.get_forecast(lat, lon, hourly=True)
        except Exception as e:
            logger.warning(f"Failed to fetch forecast for ({lat}, {lon}): {e}")
            return pd.DataFrame()

        periods = forecast.get("properties", {}).get("periods", [])

        weather_data = []
        for period in periods:
            start_time_str = period.get("startTime", "")
            if not start_time_str:
                continue

            try:
                timestamp = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
            except ValueError:
                continue

            # Make start_date and end_date timezone-aware if timestamp is
            if timestamp.tzinfo is not None:
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timestamp.tzinfo)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timestamp.tzinfo)

            if timestamp < start_date or timestamp > end_date:
                continue

            # Extract temperature
            temp = period.get("temperature", None)
            if isinstance(temp, dict):
                temp = temp.get("value", None)

            # Extract dewpoint
            dewpoint = period.get("dewpoint", {})
            if isinstance(dewpoint, dict):
                dewpoint = dewpoint.get("value", None)

            # Extract humidity
            humidity = period.get("relativeHumidity", {})
            if isinstance(humidity, dict):
                humidity = humidity.get("value", None)

            # Extract wind speed (parse "10 mph" -> 10)
            wind_speed = period.get("windSpeed", "")
            wind_speed_value = None
            if wind_speed and isinstance(wind_speed, str):
                try:
                    wind_speed_value = float(wind_speed.split()[0])
                except (ValueError, IndexError):
                    pass

            # Extract precipitation probability
            pop = period.get("probabilityOfPrecipitation", {})
            pop_value = None
            if isinstance(pop, dict):
                pop_value = pop.get("value", None)

            # Extract precipitation amount
            precip = period.get("precipitationAmount", {})
            precip_value = None
            if isinstance(precip, dict):
                precip_value = precip.get("value", None)

            weather_data.append(
                {
                    "timestamp": timestamp,
                    "lat": lat,
                    "lon": lon,
                    "temperature": temp,
                    "dewpoint": dewpoint,
                    "humidity": humidity,
                    "wind_speed": wind_speed_value,
                    "wind_direction": period.get("windDirection", None),
                    "precipitation_probability": pop_value,
                    "precipitation_amount": precip_value,
                    "short_forecast": period.get("shortForecast", ""),
                    "detailed_forecast": period.get("detailedForecast", ""),
                    "is_daytime": period.get("isDaytime", True),
                }
            )

        if not weather_data:
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        return df

    def fetch_observations_for_station(
        self, station_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical observations from a weather station
        Returns DataFrame compatible with DGXB data product schema
        """
        try:
            observations = self.get_observations(station_id, start_date, end_date)
        except Exception as e:
            logger.warning(
                f"Failed to fetch observations for station {station_id}: {e}"
            )
            return pd.DataFrame()

        features = observations.get("features", [])

        weather_data = []
        for feature in features:
            props = feature.get("properties", {})
            timestamp_str = props.get("timestamp", "")

            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Make start_date and end_date timezone-aware if timestamp is
            if timestamp.tzinfo is not None:
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timestamp.tzinfo)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timestamp.tzinfo)

            if timestamp < start_date or timestamp > end_date:
                continue

            # Extract temperature
            temp = props.get("temperature", {})
            temp_value = None
            if isinstance(temp, dict):
                temp_value = temp.get("value", None)

            # Extract dewpoint
            dewpoint = props.get("dewpoint", {})
            dewpoint_value = None
            if isinstance(dewpoint, dict):
                dewpoint_value = dewpoint.get("value", None)

            # Extract humidity
            humidity = props.get("relativeHumidity", {})
            humidity_value = None
            if isinstance(humidity, dict):
                humidity_value = humidity.get("value", None)

            # Extract wind speed
            wind_speed = props.get("windSpeed", {})
            wind_speed_value = None
            if isinstance(wind_speed, dict):
                wind_speed_value = wind_speed.get("value", None)

            # Extract wind direction
            wind_direction = props.get("windDirection", {})
            wind_direction_value = None
            if isinstance(wind_direction, dict):
                wind_direction_value = wind_direction.get("value", None)

            # Extract precipitation
            precip = props.get("precipitationLastHour", {})
            precip_value = None
            if isinstance(precip, dict):
                precip_value = precip.get("value", None)

            # Extract station location
            geometry = feature.get("geometry", {})
            coords = geometry.get("coordinates", [])
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None

            weather_data.append(
                {
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "lat": lat,
                    "lon": lon,
                    "temperature": temp_value,
                    "dewpoint": dewpoint_value,
                    "humidity": humidity_value,
                    "wind_speed": wind_speed_value,
                    "wind_direction": wind_direction_value,
                    "precipitation_amount": precip_value,
                    "text_description": props.get("textDescription", ""),
                }
            )

        if not weather_data:
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        return df

    def fetch_weather_for_locations(
        self,
        locations: List[Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        use_observations: bool = False,
        station_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch weather data for multiple locations
        Returns combined DataFrame
        """
        all_weather = []

        for lat, lon in locations:
            try:
                if use_observations and station_map and (lat, lon) in station_map:
                    station_id = station_map[(lat, lon)]
                    df = self.fetch_observations_for_station(
                        station_id, start_date, end_date
                    )
                else:
                    df = self.fetch_forecast_for_location(
                        lat, lon, start_date, end_date
                    )

                if not df.empty:
                    all_weather.append(df)
                    logger.info(f"Fetched {len(df)} weather records for ({lat}, {lon})")
            except Exception as e:
                logger.error(f"Failed to fetch weather for ({lat}, {lon}): {e}")
                continue

        if not all_weather:
            return pd.DataFrame()

        combined = pd.concat(all_weather, ignore_index=True)
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
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def fetch_and_save_weather(
    locations: List[Tuple[float, float]],
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "weather_bronze",
    use_observations: bool = False,
    station_map: Optional[dict] = None,
):
    """
    Convenience function: Fetch weather data and save to bronze-weather layer
    """
    fetcher = NWSWeatherFetcher()

    logger.info(
        f"Fetching weather data for {len(locations)} locations from {start_date} to {end_date}"
    )
    weather_df = fetcher.fetch_weather_for_locations(
        locations,
        start_date,
        end_date,
        use_observations=use_observations,
        station_map=station_map,
    )

    if weather_df.empty:
        logger.warning("No weather data fetched")
        return

    save_weather_bronze(weather_df, output_dir)
    logger.info(f"Weather data saved to {output_dir}")

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

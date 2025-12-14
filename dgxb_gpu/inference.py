"""
Inference Module with Real-Time Weather Integration

This module provides utilities for making predictions with real-time weather data.
It checks for cached weather first, then fetches from Open-Meteo API if needed.

Usage:
    from dgxb_gpu.inference import InferenceWeatherProvider, prepare_inference_features

    # Get weather for inference
    provider = InferenceWeatherProvider()
    weather = provider.get_weather_for_inference(
        h3_cells=['88283082bffffff', '88283082a7fffff'],
        target_hour=datetime.now(timezone.utc) + timedelta(hours=1)
    )

    # Prepare features with real weather
    features = prepare_inference_features(
        h3_cells=['88283082bffffff'],
        target_hour=datetime.now(timezone.utc),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WeatherCacheConfig:
    """Configuration for weather caching."""

    cache_dir: str = "weather-cache"
    cache_ttl_hours: int = 1  # How long cached data is valid
    fallback_to_historical: bool = True  # Use recent historical if forecast fails


class InferenceWeatherProvider:
    """
    Provides weather data for inference, with caching and fallback.
    Checks multiple sources:
    1. In-memory cache (fastest)
    2. File cache (fast)
    3. Open-Meteo API (network call)
    4. Historical data fallback (if forecast unavailable)
    """

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HOURLY_FIELDS = [
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "wind_speed_10m",
        "precipitation",
        "precipitation_probability",
    ]

    def __init__(self, config: Optional[WeatherCacheConfig] = None):
        self.config = config or WeatherCacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "DGXB-Inference/1.0"})
        return self._session

    def _cache_key(self, lat: float, lon: float, hour: datetime) -> str:
        """Generate cache key for a location and hour."""
        lat_str = f"{lat:.2f}".replace(".", "_").replace("-", "m")
        lon_str = f"{lon:.2f}".replace(".", "_").replace("-", "m")
        hour_str = hour.strftime("%Y%m%d_%H")
        return f"weather_{lat_str}_{lon_str}_{hour_str}"

    def _check_memory_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Check in-memory cache."""
        if key in self._memory_cache:
            cached_time, data = self._memory_cache[key]
            age_hours = (
                datetime.now(timezone.utc) - cached_time
            ).total_seconds() / 3600
            if age_hours < self.config.cache_ttl_hours:
                logger.debug(f"Memory cache hit: {key}")
                return data
            else:
                del self._memory_cache[key]
        return None

    def _check_file_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Check file cache."""
        cache_path = self.cache_dir / f"{key}.parquet"
        if cache_path.exists():
            try:
                # Check file age
                file_age_hours = (
                    datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                ).total_seconds() / 3600

                if file_age_hours < self.config.cache_ttl_hours:
                    df = pd.read_parquet(cache_path)
                    logger.debug(f"File cache hit: {cache_path}")
                    # Also store in memory cache
                    self._memory_cache[key] = (datetime.now(timezone.utc), df)
                    return df
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, key: str, df: pd.DataFrame):
        """Save to both memory and file cache."""
        self._memory_cache[key] = (datetime.now(timezone.utc), df)
        cache_path = self.cache_dir / f"{key}.parquet"
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def _fetch_forecast(
        self,
        lat: float,
        lon: float,
        target_hour: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch forecast from Open-Meteo API."""
        import time

        # Ensure timezone
        if target_hour.tzinfo is None:
            target_hour = target_hour.replace(tzinfo=timezone.utc)

        # Fetch a small window around target hour
        start_date = (target_hour - timedelta(hours=2)).strftime("%Y-%m-%d")
        end_date = (target_hour + timedelta(hours=24)).strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HOURLY_FIELDS),
            "timezone": "UTC",
        }

        try:
            time.sleep(0.1)  # Rate limiting
            resp = self.session.get(self.FORECAST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Forecast API failed for ({lat}, {lon}): {e}")
            return None

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(times, utc=True),
                "lat": lat,
                "lon": lon,
                "weather_temperature": hourly.get(
                    "temperature_2m", [None] * len(times)
                ),
                "weather_humidity": hourly.get(
                    "relative_humidity_2m", [None] * len(times)
                ),
                "weather_dewpoint": hourly.get("dewpoint_2m", [None] * len(times)),
                "weather_wind_speed": hourly.get("wind_speed_10m", [None] * len(times)),
                "weather_precipitation_amount": hourly.get(
                    "precipitation", [None] * len(times)
                ),
                "weather_precipitation_probability": hourly.get(
                    "precipitation_probability", [None] * len(times)
                ),
                "data_source": "forecast",
            }
        )

        # Convert Celsius to Fahrenheit
        if "weather_temperature" in df.columns:
            df["weather_temperature"] = df["weather_temperature"] * 9 / 5 + 32
        if "weather_dewpoint" in df.columns:
            df["weather_dewpoint"] = df["weather_dewpoint"] * 9 / 5 + 32

        return df

    def get_weather_for_location(
        self,
        lat: float,
        lon: float,
        target_hour: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Get weather for a specific location and hour.
        Checks cache first, then fetches from API.
        """
        if target_hour.tzinfo is None:
            target_hour = target_hour.replace(tzinfo=timezone.utc)

        target_hour_floor = target_hour.replace(minute=0, second=0, microsecond=0)
        cache_key = self._cache_key(lat, lon, target_hour_floor)

        # Check caches
        cached = self._check_memory_cache(cache_key)
        if cached is not None:
            return cached

        cached = self._check_file_cache(cache_key)
        if cached is not None:
            return cached

        # Fetch from API
        logger.info(
            f"Fetching forecast for ({lat:.2f}, {lon:.2f}) at {target_hour_floor}"
        )
        df = self._fetch_forecast(lat, lon, target_hour)

        if df is not None and len(df) > 0:
            # Filter to target hour
            df_hour = df[df["timestamp"] == target_hour_floor]
            if len(df_hour) > 0:
                self._save_to_cache(cache_key, df_hour)
                return df_hour

            # If exact hour not found, use nearest
            df["time_diff"] = abs(df["timestamp"] - target_hour_floor)
            nearest = df.loc[df["time_diff"].idxmin()].to_frame().T
            nearest = nearest.drop(columns=["time_diff"])
            self._save_to_cache(cache_key, nearest)
            return nearest

        return None

    def get_weather_for_inference(
        self,
        h3_cells: List[str],
        target_hour: datetime,
    ) -> pd.DataFrame:
        """
        Get weather for multiple H3 cells at a target hour.
        Returns a DataFrame with weather for each cell.
        """
        try:
            import h3
        except ImportError:
            raise ImportError("h3 library required: pip install h3")

        if target_hour.tzinfo is None:
            target_hour = target_hour.replace(tzinfo=timezone.utc)

        # Group cells by rounded location to reduce API calls
        locations: Dict[Tuple[float, float], List[str]] = {}

        for cell in h3_cells:
            try:
                lat, lon = h3.cell_to_latlng(cell)
            except AttributeError:
                lat, lon = h3.h3_to_geo(cell)

            # Round to reduce API calls
            lat_round = round(lat, 2)
            lon_round = round(lon, 2)
            key = (lat_round, lon_round)

            if key not in locations:
                locations[key] = []
            locations[key].append(cell)

        # Fetch weather for each unique location
        all_weather = []

        for (lat, lon), cells in locations.items():
            weather_df = self.get_weather_for_location(lat, lon, target_hour)

            if weather_df is not None and len(weather_df) > 0:
                for cell in cells:
                    cell_weather = weather_df.copy()
                    cell_weather["h3_cell"] = cell
                    cell_weather["hour_ts"] = target_hour.replace(
                        minute=0, second=0, microsecond=0
                    )
                    all_weather.append(cell_weather)

        if not all_weather:
            logger.warning("No weather data available for inference")
            return pd.DataFrame()

        result = pd.concat(all_weather, ignore_index=True)
        return result

    def check_weather_available(
        self,
        h3_cells: List[str],
        target_hour: datetime,
    ) -> Dict[str, bool]:
        """
        Check if weather data is available for cells without fetching.
        Returns dict mapping cell -> availability status.
        """
        try:
            import h3
        except ImportError:
            return {cell: False for cell in h3_cells}

        if target_hour.tzinfo is None:
            target_hour = target_hour.replace(tzinfo=timezone.utc)

        target_hour_floor = target_hour.replace(minute=0, second=0, microsecond=0)
        availability = {}

        for cell in h3_cells:
            try:
                lat, lon = h3.cell_to_latlng(cell)
            except AttributeError:
                lat, lon = h3.h3_to_geo(cell)

            lat_round = round(lat, 2)
            lon_round = round(lon, 2)
            cache_key = self._cache_key(lat_round, lon_round, target_hour_floor)

            # Check caches
            in_memory = cache_key in self._memory_cache
            cache_path = self.cache_dir / f"{cache_key}.parquet"
            in_file = cache_path.exists()

            availability[cell] = in_memory or in_file

        return availability


def prepare_inference_features(
    h3_cells: List[str],
    target_hour: datetime,
    include_temporal: bool = True,
    weather_provider: Optional[InferenceWeatherProvider] = None,
) -> pd.DataFrame:
    """
    Prepare feature DataFrame for inference with real-time weather.

    Args:
        h3_cells: List of H3 cell IDs to predict for
        target_hour: The hour to predict (prediction made FOR this hour)
        include_temporal: Include temporal features (hour, day_of_week, etc.)
        weather_provider: Optional weather provider instance (creates new if None)

    Returns:
        DataFrame ready for model prediction with weather features
    """
    if target_hour.tzinfo is None:
        target_hour = target_hour.replace(tzinfo=timezone.utc)

    target_hour_floor = target_hour.replace(minute=0, second=0, microsecond=0)

    # Initialize weather provider
    if weather_provider is None:
        weather_provider = InferenceWeatherProvider()

    # Fetch weather
    weather_df = weather_provider.get_weather_for_inference(h3_cells, target_hour_floor)

    # Build features DataFrame
    if len(weather_df) > 0:
        features = weather_df[["h3_cell", "hour_ts"]].copy()

        # Add weather columns
        weather_cols = [
            "weather_temperature",
            "weather_humidity",
            "weather_dewpoint",
            "weather_wind_speed",
            "weather_precipitation_amount",
            "weather_precipitation_probability",
        ]
        for col in weather_cols:
            if col in weather_df.columns:
                features[col] = weather_df[col].values
    else:
        # Create basic structure without weather
        features = pd.DataFrame(
            {
                "h3_cell": h3_cells,
                "hour_ts": [target_hour_floor] * len(h3_cells),
            }
        )
        logger.warning("No weather data available, features will have missing weather")

    # Add temporal features
    if include_temporal:
        features["hour"] = target_hour_floor.hour
        features["day_of_week"] = target_hour_floor.weekday()
        features["day_of_month"] = target_hour_floor.day
        features["month"] = target_hour_floor.month
        features["is_weekend"] = int(target_hour_floor.weekday() >= 5)

        # Cyclical encodings
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    # Add weather availability flag
    features["has_weather_data"] = int(len(weather_df) > 0)

    return features


def batch_inference_with_weather(
    model,
    h3_cells: List[str],
    target_hours: List[datetime],
    feature_columns: List[str],
    weather_provider: Optional[InferenceWeatherProvider] = None,
) -> pd.DataFrame:
    """
    Run batch inference for multiple cells and hours with real-time weather.

    Args:
        model: Trained model with predict() method
        h3_cells: List of H3 cells to predict for
        target_hours: List of hours to predict for
        feature_columns: Columns expected by model (in order)
        weather_provider: Optional weather provider instance

    Returns:
        DataFrame with predictions for each (cell, hour) combination
    """
    if weather_provider is None:
        weather_provider = InferenceWeatherProvider()

    all_predictions = []

    for target_hour in target_hours:
        # Prepare features
        features = prepare_inference_features(
            h3_cells=h3_cells,
            target_hour=target_hour,
            include_temporal=True,
            weather_provider=weather_provider,
        )

        # Ensure feature order matches model expectations
        missing_cols = set(feature_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0  # Fill missing with zeros

        # Select and order columns
        X = features[feature_columns].values

        # Make predictions
        try:
            predictions = model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed for {target_hour}: {e}")
            predictions = [np.nan] * len(h3_cells)

        # Build result
        result = pd.DataFrame(
            {
                "h3_cell": h3_cells,
                "target_hour": target_hour,
                "prediction": predictions,
                "has_weather": (
                    features["has_weather_data"].values
                    if "has_weather_data" in features.columns
                    else 1
                ),
            }
        )
        all_predictions.append(result)

    return pd.concat(all_predictions, ignore_index=True)


class IncidentPredictionService:
    """
    Production-ready prediction service with real-time weather integration.

    This service:
    1. Loads a trained model from disk
    2. Fetches real-time weather from Open-Meteo API
    3. Makes predictions with weather features

    Usage:
        service = IncidentPredictionService.from_training_results("results/models")
        predictions = service.predict(
            h3_cells=["88283082bffffff"],
            target_hour=datetime.now(timezone.utc) + timedelta(hours=1)
        )
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        model_name: str = "unknown",
        weather_provider: Optional[InferenceWeatherProvider] = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.model_name = model_name
        self.weather_provider = weather_provider or InferenceWeatherProvider()

        # Identify weather features
        self.weather_features = [
            col for col in feature_columns if "weather" in col.lower()
        ]
        logger.info(
            f"Initialized prediction service with {len(feature_columns)} features "
            f"({len(self.weather_features)} weather features)"
        )

    @classmethod
    def from_training_results(
        cls,
        models_dir: str = "results/models",
        channel: str = "rich",
    ) -> "IncidentPredictionService":
        """
        Load prediction service from training results.

        Args:
            models_dir: Path to models directory
            channel: Which channel to load ("base" or "rich")

        Returns:
            Configured IncidentPredictionService
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required: pip install joblib")

        models_path = Path(models_dir)

        # Load metadata
        metadata_path = models_path / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Run training pipeline first."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        if channel not in metadata.get("models", {}):
            available = list(metadata.get("models", {}).keys())
            raise ValueError(f"Channel '{channel}' not found. Available: {available}")

        model_info = metadata["models"][channel]

        # Load model
        model_path = models_path / f"champion_{channel}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        return cls(
            model=model,
            feature_columns=model_info["feature_columns"],
            model_name=model_info["model_name"],
        )

    def predict(
        self,
        h3_cells: List[str],
        target_hour: datetime,
        fetch_weather: bool = True,
    ) -> pd.DataFrame:
        """
        Make predictions for given cells and hour.

        Args:
            h3_cells: List of H3 cell IDs to predict for
            target_hour: Hour to predict incidents for
            fetch_weather: Whether to fetch real-time weather (default True)

        Returns:
            DataFrame with predictions and metadata
        """
        if target_hour.tzinfo is None:
            target_hour = target_hour.replace(tzinfo=timezone.utc)

        target_hour_floor = target_hour.replace(minute=0, second=0, microsecond=0)

        # Prepare features with weather
        if fetch_weather and self.weather_features:
            features = prepare_inference_features(
                h3_cells=h3_cells,
                target_hour=target_hour_floor,
                include_temporal=True,
                weather_provider=self.weather_provider,
            )
        else:
            # No weather fetch - use zeros for weather features
            features = prepare_inference_features(
                h3_cells=h3_cells,
                target_hour=target_hour_floor,
                include_temporal=True,
                weather_provider=None,
            )

        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0

        # Select and order columns
        X = features[self.feature_columns].values.astype(np.float32)

        # Make predictions
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # No negative incidents

        # Build result
        result = pd.DataFrame(
            {
                "h3_cell": h3_cells,
                "target_hour": target_hour_floor,
                "predicted_incidents": predictions,
                "has_weather_data": features.get("has_weather_data", 1),
                "model_name": self.model_name,
                "prediction_timestamp": datetime.now(timezone.utc),
            }
        )

        # Add weather values if available
        for col in self.weather_features:
            if col in features.columns:
                result[col] = features[col].values

        return result

    def predict_next_hours(
        self,
        h3_cells: List[str],
        hours_ahead: int = 24,
        start_hour: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Predict incidents for the next N hours.

        Args:
            h3_cells: List of H3 cell IDs
            hours_ahead: Number of hours to predict (default 24)
            start_hour: Starting hour (default: current hour)

        Returns:
            DataFrame with predictions for all hours
        """
        if start_hour is None:
            start_hour = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
        elif start_hour.tzinfo is None:
            start_hour = start_hour.replace(tzinfo=timezone.utc)

        target_hours = [start_hour + timedelta(hours=i) for i in range(hours_ahead)]

        all_predictions = []
        for target_hour in target_hours:
            pred = self.predict(h3_cells, target_hour)
            all_predictions.append(pred)

        return pd.concat(all_predictions, ignore_index=True)

    def get_hotspots(
        self,
        h3_cells: List[str],
        target_hour: datetime,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Get top-K predicted incident hotspots.

        Args:
            h3_cells: List of H3 cells to consider
            target_hour: Hour to predict for
            top_k: Number of top hotspots to return

        Returns:
            DataFrame with top-K hotspots sorted by predicted incidents
        """
        predictions = self.predict(h3_cells, target_hour)
        return (
            predictions.sort_values("predicted_incidents", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example H3 cells (Long Beach, CA area)
    test_cells = ["88283082bffffff", "88283082a7fffff"]
    target = datetime.now(timezone.utc) + timedelta(hours=1)

    print("=" * 60)
    print("DGXB Inference Module - Real-Time Weather Integration")
    print("=" * 60)

    # 1. Test weather provider
    print("\n[1] Testing weather provider...")
    provider = InferenceWeatherProvider()
    weather = provider.get_weather_for_inference(test_cells, target)
    print(f"    Fetched {len(weather)} weather records for {len(test_cells)} cells")

    if len(weather) > 0:
        print(f"    Weather columns: {list(weather.columns)}")
        print(
            f"    Temperature range: {weather['weather_temperature'].min():.1f} - {weather['weather_temperature'].max():.1f} F"
        )

    # 2. Prepare inference features
    print("\n[2] Preparing inference features with weather...")
    features = prepare_inference_features(test_cells, target)
    print(f"    Features shape: {features.shape}")
    print(
        f"    Weather features: {[c for c in features.columns if 'weather' in c.lower()]}"
    )

    # 3. Try loading prediction service (if model exists)
    print("\n[3] Loading prediction service...")
    try:
        service = IncidentPredictionService.from_training_results("results/models")
        print(f"    Model loaded: {service.model_name}")
        print(f"    Feature count: {len(service.feature_columns)}")
        print(f"    Weather features: {len(service.weather_features)}")

        # Make prediction
        print("\n[4] Making predictions with real-time weather...")
        predictions = service.predict(test_cells, target)
        print(f"    Predictions for {len(predictions)} cells:")
        print(
            predictions[
                ["h3_cell", "target_hour", "predicted_incidents", "has_weather_data"]
            ].to_string(index=False)
        )

        # Get hotspots
        print("\n[5] Getting top hotspots...")
        hotspots = service.get_hotspots(test_cells, target, top_k=5)
        print(f"    Top {len(hotspots)} hotspots by predicted incidents")

    except FileNotFoundError as e:
        print(f"    Model not found: {e}")
        print("    Run training pipeline first: python -m dgxb_gpu.run_training")

    print("\n" + "=" * 60)
    print("Usage in production:")
    print("  from dgxb_gpu.inference import IncidentPredictionService")
    print(
        "  service = IncidentPredictionService.from_training_results('results/models')"
    )
    print("  predictions = service.predict(h3_cells, target_hour)")
    print("=" * 60)

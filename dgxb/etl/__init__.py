"""
ETL pipelines for DGXB
Includes CPU and GPU ingestion, feature building, and data fetching
"""

from .weather_fetcher import (
    NWSWeatherFetcher,
    fetch_and_save_weather,
    save_weather_bronze,
)
from .traffic_fetcher import (
    AustinTrafficFetcher,
    fetch_and_save_traffic,
    save_traffic_bronze,
)
from .silver_processor import (
    process_bronze_to_silver_traffic,
    process_bronze_to_silver_weather,
    load_bronze_parquets,
    clean_traffic_data,
    clean_weather_data,
)

__all__ = [
    "NWSWeatherFetcher",
    "fetch_and_save_weather",
    "save_weather_bronze",
    "AustinTrafficFetcher",
    "fetch_and_save_traffic",
    "save_traffic_bronze",
    "process_bronze_to_silver_traffic",
    "process_bronze_to_silver_weather",
    "load_bronze_parquets",
    "clean_traffic_data",
    "clean_weather_data",
]

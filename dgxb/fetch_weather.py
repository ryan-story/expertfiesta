#!/usr/bin/env python3
"""
Quick script to fetch weather data and save to weather_bronze
Usage: python fetch_weather.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from etl.weather_fetcher import fetch_and_save_weather

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Example: Austin, TX area locations
    locations = [
        (30.2672, -97.7431),  # Downtown Austin
        (30.3072, -97.7559),  # North Austin
        (30.2244, -97.7694),  # South Austin
    ]
    
    # Fetch last 7 days of forecast data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Fetching weather data for {len(locations)} locations")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output: weather_bronze/")
    
    weather_df = fetch_and_save_weather(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        output_dir="weather_bronze"
    )
    
    if weather_df is not None and not weather_df.empty:
        print(f"\nSuccess! Fetched {len(weather_df)} weather records")
        print(f"Columns: {list(weather_df.columns)}")
        print(f"Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    else:
        print("\nNo weather data fetched. Check logs for errors.")

if __name__ == "__main__":
    main()


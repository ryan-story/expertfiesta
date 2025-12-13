"""
Austin Traffic Incident Data Fetcher for DGXB
Pulls traffic incident data from Austin, Texas Open Data Portal
API: https://data.austintexas.gov/api/v3/views/dx9v-zd7x/query.json
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class AustinTrafficFetcher:
    """
    Fetches traffic incident data from Austin, Texas Open Data Portal
    """

    BASE_URL = "https://data.austintexas.gov/resource/dx9v-zd7x.json"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "DGXB/1.0 (transportation-intelligence@example.com)",
                "Accept": "application/json",
            }
        )

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request"""
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def fetch_incidents(
        self,
        limit: int = 1000,
        offset: int = 0,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch traffic incidents from Austin API (Socrata format)
        Returns DataFrame compatible with DGXB data product schema
        """
        params = {"$limit": limit, "$offset": offset}

        if where_clause:
            params["$where"] = where_clause

        if order_by:
            params["$order"] = order_by

        logger.info(f"Fetching traffic incidents: limit={limit}, offset={offset}")
        data = self._make_request(params)

        if not data:
            logger.warning("No data returned from API")
            return pd.DataFrame()

        # Socrata API returns list of records directly
        if not isinstance(data, list):
            logger.warning(f"Unexpected data format: {type(data)}")
            return pd.DataFrame()

        if not data:
            logger.warning("Empty records list")
            return pd.DataFrame()

        # Create DataFrame directly from records
        df = pd.DataFrame(data)

        # Standardize column names (common Austin data portal columns)
        column_mapping = {
            "incident_number": "incident_id",
            "traffic_report_id": "incident_id",
            "published_date": "timestamp",
            "published_datetime": "timestamp",
            "issue_reported": "description",
            "latitude": "lat",
            "longitude": "lon",
            "traffic_report_status": "status",
        }

        # Rename columns if they exist (avoid duplicates)
        rename_dict = {}
        for k, v in column_mapping.items():
            if k in df.columns and v not in df.columns:
                rename_dict[k] = v

        df = df.rename(columns=rename_dict)

        # Try to parse timestamp if it exists (only if not already datetime)
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                try:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], errors="coerce", utc=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp: {e}")

        # Ensure lat/lon are numeric
        if "lat" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        if "lon" in df.columns:
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

        # Handle location.geometry if present
        if "location" in df.columns:
            # Extract coordinates from location.geometry if it's a dict
            def extract_coords(loc):
                if isinstance(loc, dict) and "coordinates" in loc:
                    coords = loc["coordinates"]
                    if len(coords) >= 2:
                        return pd.Series({"lon": coords[0], "lat": coords[1]})
                return pd.Series({"lon": None, "lat": None})

            if df["location"].apply(lambda x: isinstance(x, dict)).any():
                coords_df = df["location"].apply(extract_coords)
                if "lon" not in df.columns or df["lon"].isna().all():
                    df["lon"] = coords_df["lon"]
                if "lat" not in df.columns or df["lat"].isna().all():
                    df["lat"] = coords_df["lat"]

        logger.info(f"Fetched {len(df)} traffic incident records")
        return df

    def fetch_recent_incidents(self, days: int = 7, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch recent traffic incidents (last N days)
        """
        from datetime import timezone

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Order by date descending
        order_by = "published_date DESC"

        # Try without date filter first, then filter in pandas
        df = self.fetch_incidents(limit=limit, order_by=order_by)

        if df.empty:
            return df

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Find timestamp column (already renamed in fetch_incidents)
        if "timestamp" in df.columns:
            # Parse and filter by date (handle timezone)
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce", utc=True
                )

            # Make start_date and end_date timezone-aware if timestamp is
            if df["timestamp"].dt.tz is not None:
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)

            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

        return df


def save_traffic_bronze(
    traffic_df: pd.DataFrame,
    output_dir: str = "bronze-traffic",
    partition_by_date: bool = True,
):
    """
    Save traffic incident data to bronze layer (raw ingested data)
    Follows data lakehouse bronze/silver/gold pattern
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if traffic_df.empty:
        logger.warning("Empty traffic DataFrame, nothing to save")
        return

    if partition_by_date and "timestamp" in traffic_df.columns:
        # Partition by date for efficient querying
        traffic_df = traffic_df.copy()
        date_col_created = False

        # Ensure timestamp is datetime (already parsed in fetch_recent_incidents)
        if pd.api.types.is_datetime64_any_dtype(traffic_df["timestamp"]):
            traffic_df["date"] = traffic_df["timestamp"].dt.date
            date_col_created = True
        else:
            # If not datetime, try to convert
            try:
                traffic_df["timestamp"] = pd.to_datetime(
                    traffic_df["timestamp"], errors="coerce"
                )
                traffic_df["date"] = traffic_df["timestamp"].dt.date
                date_col_created = True
            except Exception as e:
                logger.warning(f"Could not partition by date: {e}")
                partition_by_date = False

        if partition_by_date and date_col_created:
            for date, group_df in traffic_df.groupby("date"):
                date_str = date.strftime("%Y-%m-%d")
                file_path = output_path / f"traffic_{date_str}.parquet"
                group_df.drop(columns=["date"]).to_parquet(file_path, index=False)
                logger.info(f"Saved {len(group_df)} records to {file_path}")
            return
    else:
        # Single file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"traffic_{timestamp_str}.parquet"
        traffic_df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(traffic_df)} records to {file_path}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(traffic_df),
        "columns": list(traffic_df.columns),
        "date_range": {
            "start": (
                traffic_df["timestamp"].min().isoformat()
                if "timestamp" in traffic_df.columns
                and pd.api.types.is_datetime64_any_dtype(traffic_df["timestamp"])
                and not traffic_df["timestamp"].isna().all()
                else None
            ),
            "end": (
                traffic_df["timestamp"].max().isoformat()
                if "timestamp" in traffic_df.columns
                and pd.api.types.is_datetime64_any_dtype(traffic_df["timestamp"])
                and not traffic_df["timestamp"].isna().all()
                else None
            ),
        },
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def fetch_and_save_traffic(
    days: int = 7, limit: int = 10000, output_dir: str = "bronze-traffic"
):
    """
    Convenience function: Fetch traffic incident data and save to bronze layer
    """
    fetcher = AustinTrafficFetcher()

    logger.info(f"Fetching traffic incidents from last {days} days (limit: {limit})")
    traffic_df = fetcher.fetch_recent_incidents(days=days, limit=limit)

    if traffic_df.empty:
        logger.warning("No traffic incident data fetched")
        return

    save_traffic_bronze(traffic_df, output_dir)
    logger.info(f"Traffic incident data saved to {output_dir}")

    return traffic_df


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Fetch last 7 days of traffic incidents
    fetch_and_save_traffic(days=7, limit=5000)

"""
Austin Traffic Incident Data Fetcher for DGXB (GPU)
Pulls traffic incident data from Austin, Texas Open Data Portal (Socrata)
API: https://data.austintexas.gov/resource/dx9v-zd7x.json

Notes:
- Network I/O (requests) remains CPU-bound.
- Post-fetch transforms are GPU-accelerated with cuDF.
- Output bronze files remain parquet, partitioned by date.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # CPU helper: timestamp conversions for API calls
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
class FetchParams:
    limit: int = 1000
    offset: int = 0
    where_clause: Optional[str] = None
    order_by: Optional[str] = None


@dataclass(frozen=True)
class BronzeSaveParams:
    output_dir: str = "bronze-traffic"
    partition_by_date: bool = True


class AustinTrafficFetcherGPU:
    """
    Fetches traffic incident data from Austin, Texas Open Data Portal.
    GPU-accelerated transformations after fetch.
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

    def _make_request(self, params: Dict[str, Any]) -> Any:
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    @staticmethod
    def _records_to_cudf(records: List[Dict[str, Any]]):
        """
        Convert list-of-dict records to cuDF DataFrame efficiently.
        Uses pyarrow.Table for fewer type surprises than direct cudf.DataFrame(records).
        """
        _require_gpu_libs()
        import cudf
        import pyarrow as pa

        if not records:
            return cudf.DataFrame()

        # Arrow handles nested structs better than direct cudf.DataFrame(list_of_dict)
        table = pa.Table.from_pylist(records)
        gdf = cudf.DataFrame.from_arrow(table)
        return gdf

    @staticmethod
    def _coerce_numeric(gdf, col: str):
        import cudf

        if col not in gdf.columns:
            return gdf
        try:
            gdf[col] = cudf.to_numeric(gdf[col], errors="coerce")
        except Exception:
            gdf[col] = cudf.to_numeric(gdf[col].astype("str"), errors="coerce")
        return gdf

    @staticmethod
    def _parse_timestamp_utc(gdf, col: str = "timestamp"):
        import cudf

        if col not in gdf.columns:
            return gdf
        try:
            gdf[col] = cudf.to_datetime(gdf[col], utc=True, errors="coerce")
        except Exception:
            # older cuDF may not accept errors
            gdf[col] = cudf.to_datetime(gdf[col], utc=True)
        return gdf

    @staticmethod
    def _flatten_location(gdf):
        """
        Flatten 'location' if present, extracting lon/lat.
        Supports struct-like 'location' with fields:
          - type
          - coordinates (list [lon, lat])
        If 'location' is JSON string and cuDF supports json_extract, it will use it.
        """

        if "location" not in gdf.columns:
            return gdf

        gdf = gdf.copy()
        loc = gdf["location"]

        # Ensure target cols exist
        if "location_type" not in gdf.columns:
            gdf["location_type"] = None
        if "location_lon" not in gdf.columns:
            gdf["location_lon"] = None
        if "location_lat" not in gdf.columns:
            gdf["location_lat"] = None

        # Struct path
        try:
            if hasattr(loc, "struct"):
                try:
                    gdf["location_type"] = loc.struct.field("type")
                except Exception:
                    pass
                try:
                    coords = loc.struct.field("coordinates")
                    if hasattr(coords, "list"):
                        gdf["location_lon"] = coords.list.get(0)
                        gdf["location_lat"] = coords.list.get(1)
                except Exception:
                    pass

                gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "location_lon")
                gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "location_lat")
                return gdf
        except Exception:
            pass

        # JSON string path (GPU if supported)
        try:
            s = loc.astype("str")
            gdf["location_type"] = s.str.json_extract("$.type")
            gdf["location_lon"] = s.str.json_extract("$.coordinates[0]")
            gdf["location_lat"] = s.str.json_extract("$.coordinates[1]")
            gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "location_lon")
            gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "location_lat")
            return gdf
        except Exception:
            # No GPU json_extract support; do nothing (lat/lon may already exist)
            return gdf

    @staticmethod
    def _standardize_schema(gdf):
        """
        Standardize column names and ensure expected DGXB schema fields.
        Avoid duplicates.
        """
        if gdf is None or len(gdf) == 0:
            return gdf

        # Common Austin/Socrata column mapping
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

        # Rename only when source exists AND target not already present
        rename_dict = {}
        for src, tgt in column_mapping.items():
            if src in gdf.columns and tgt not in gdf.columns:
                rename_dict[src] = tgt

        if rename_dict:
            gdf = gdf.rename(columns=rename_dict)

        # Remove duplicate columns if any
        cols = list(gdf.columns)
        seen = set()
        keep = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                keep.append(c)
        if len(keep) != len(cols):
            gdf = gdf[keep]

        # Timestamp parse
        gdf = AustinTrafficFetcherGPU._parse_timestamp_utc(gdf, "timestamp")

        # Numeric lat/lon
        gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "lat")
        gdf = AustinTrafficFetcherGPU._coerce_numeric(gdf, "lon")

        # Flatten location and use it to fill missing lat/lon if needed
        gdf = AustinTrafficFetcherGPU._flatten_location(gdf)

        if "lat" in gdf.columns and "location_lat" in gdf.columns:
            gdf["lat"] = gdf["lat"].fillna(gdf["location_lat"])
        if "lon" in gdf.columns and "location_lon" in gdf.columns:
            gdf["lon"] = gdf["lon"].fillna(gdf["location_lon"])

        return gdf

    def fetch_incidents_gpu(
        self,
        limit: int = 1000,
        offset: int = 0,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
    ):
        """
        Fetch traffic incidents (single page) and return cuDF DataFrame.
        """
        _require_gpu_libs()

        params: Dict[str, Any] = {"$limit": limit, "$offset": offset}
        if where_clause:
            params["$where"] = where_clause
        if order_by:
            params["$order"] = order_by

        logger.info(f"Fetching traffic incidents: limit={limit}, offset={offset}")
        data = self._make_request(params)

        if not data or not isinstance(data, list):
            logger.warning(f"No data or unexpected data format: {type(data)}")
            from cudf import DataFrame  # type: ignore

            return DataFrame()

        gdf = self._records_to_cudf(data)
        gdf = self._standardize_schema(gdf)

        logger.info(f"Fetched {len(gdf)} traffic incident records (GPU DF)")
        return gdf

    def fetch_recent_incidents_gpu(
        self,
        days: int = 7,
        limit: int = 10000,
        page_size: int = 2000,
        order_by: str = "published_date DESC",
    ):
        """
        Fetch recent traffic incidents (last N days), returning cuDF DF.
        Uses pagination to reliably pull up to `limit` records.
        Filters by timestamp on GPU after fetch (fast and robust vs Socrata where quirks).

        Args:
          days: lookback window
          limit: max records to return (best-effort)
          page_size: page size for Socrata pagination (<= 50000, typical safe 2000-10000)
          order_by: Socrata order clause
        """
        _require_gpu_libs()
        import cudf

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Pull pages until we hit limit or API returns empty
        all_pages = []
        offset = 0

        while offset < limit:
            batch_limit = min(page_size, limit - offset)
            gdf = self.fetch_incidents_gpu(
                limit=batch_limit,
                offset=offset,
                where_clause=None,
                order_by=order_by,
            )
            if gdf is None or len(gdf) == 0:
                break
            all_pages.append(gdf)
            offset += batch_limit

            # Early stop if the last timestamp in this page is already older than start_date
            if "timestamp" in gdf.columns:
                try:
                    tmin = gdf["timestamp"].min()
                    if tmin is not None:
                        # convert minimal scalar to pandas Timestamp
                        tmin_pd = pd.Timestamp(tmin.to_pandas())
                        if tmin_pd < start_date:
                            # We likely crossed beyond window in descending order
                            break
                except Exception:
                    pass

        if not all_pages:
            return cudf.DataFrame()

        gdf_all = cudf.concat(all_pages, ignore_index=True)

        # Filter by date window on GPU
        if "timestamp" in gdf_all.columns:
            # Convert python datetimes to pandas timestamps for consistent comparison
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            gdf_all = gdf_all[
                (gdf_all["timestamp"] >= start_ts) & (gdf_all["timestamp"] <= end_ts)
            ]

        # De-dupe by incident_id if present (Socrata can repeat)
        if "incident_id" in gdf_all.columns:
            gdf_all = gdf_all.drop_duplicates(subset=["incident_id"])

        return gdf_all


def save_traffic_bronze_gpu(
    gdf,
    output_dir: str = "bronze-traffic",
    partition_by_date: bool = True,
):
    """
    Save traffic incident data to bronze layer (raw ingested data) using GPU write.
    Partitions by date if enabled.
    """
    _require_gpu_libs()
    import cudf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if gdf is None or len(gdf) == 0:
        logger.warning("Empty traffic DataFrame, nothing to save")
        return

    # Partition by date using GPU datetime accessor, then write file-per-date
    if partition_by_date and "timestamp" in gdf.columns:
        gdf = gdf.copy()

        # Create date column (as string YYYY-MM-DD for easy groupby + naming)
        try:
            # normalize to date via pandas-style dt.floor and dt.strftime
            # cuDF supports dt.strftime in most versions
            gdf["date_str"] = gdf["timestamp"].dt.strftime("%Y-%m-%d")
        except Exception:
            # fallback: move minimal work to CPU for the date string only
            pdf_dates = gdf[["timestamp"]].to_pandas()
            pdf_dates["date_str"] = pd.to_datetime(
                pdf_dates["timestamp"], utc=True
            ).dt.strftime("%Y-%m-%d")
            gdf["date_str"] = cudf.from_pandas(pdf_dates["date_str"])

        # Write per date
        for date_str, group in gdf.groupby("date_str"):
            file_path = output_path / f"traffic_{date_str}.parquet"
            group = group.drop(columns=["date_str"])
            group.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(group)} records to {file_path}")
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"traffic_{timestamp_str}.parquet"
        gdf.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(gdf)} records to {file_path}")

    # Metadata (minimal CPU transfer)
    date_start = None
    date_end = None
    if "timestamp" in gdf.columns:
        try:
            tmin = gdf["timestamp"].min()
            tmax = gdf["timestamp"].max()
            if tmin is not None and tmax is not None:
                date_start = pd.Timestamp(tmin.to_pandas()).isoformat()
                date_end = pd.Timestamp(tmax.to_pandas()).isoformat()
        except Exception:
            pass

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": int(len(gdf)),
        "columns": list(gdf.columns),
        "date_range": {"start": date_start, "end": date_end},
        "engine": "cudf",
        "source": "austin_socrata_dx9v-zd7x",
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def fetch_and_save_traffic_gpu(
    days: int = 7,
    limit: int = 10000,
    output_dir: str = "bronze-traffic",
    page_size: int = 2000,
):
    """
    Convenience function: Fetch traffic incidents and save to bronze layer (GPU write).
    """
    fetcher = AustinTrafficFetcherGPU()

    logger.info(f"Fetching traffic incidents from last {days} days (limit: {limit})")
    gdf = fetcher.fetch_recent_incidents_gpu(
        days=days, limit=limit, page_size=page_size
    )

    if gdf is None or len(gdf) == 0:
        logger.warning("No traffic incident data fetched")
        return None

    save_traffic_bronze_gpu(gdf, output_dir=output_dir, partition_by_date=True)
    logger.info(f"Traffic incident data saved to {output_dir}")
    return gdf


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Fetch last 7 days of traffic incidents
    fetch_and_save_traffic_gpu(
        days=7, limit=5000, output_dir="bronze-traffic", page_size=2000
    )

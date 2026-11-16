"""Shared helpers for the Montandon Cookbook notebooks.

The notebooks import this module so that connection handling, paging, flattening
of STAC items and hazard classification are written (and fixed) in one place.

Configuration (environment variables):

* ``MONTANDON_API_TOKEN`` - Bearer token from the IFRC GO platform (required for the
  landing page; prompted for interactively when missing).
* ``MONTANDON_API_URL``   - API root. Defaults to **production**
  (``https://montandon-eoapi.ifrc.org/stac``). Use the staging URL
  (``https://montandon-eoapi-stage.ifrc.org/stac``) only for testing.
"""

from __future__ import annotations

import json
import os
import time
from getpass import getpass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

PRODUCTION_URL = "https://montandon-eoapi.ifrc.org/stac"
STAGING_URL = "https://montandon-eoapi-stage.ifrc.org/stac"
API_URL = os.getenv("MONTANDON_API_URL", PRODUCTION_URL).rstrip("/")

CACHE_DIR = Path(os.getenv("MONTANDON_CACHE_DIR", Path.home() / ".cache" / "montandon-cookbook"))

# ---------------------------------------------------------------------------
# Authentication & connection
# ---------------------------------------------------------------------------


def get_token() -> str:
    """Return the API token from ``MONTANDON_API_TOKEN`` or prompt for it once."""
    token = os.getenv("MONTANDON_API_TOKEN", "").strip()
    if not token:
        token = getpass("Montandon API token (from https://goadmin.ifrc.org/): ").strip()
        if not token:
            raise ValueError("A Montandon API token is required. See the Getting Started page.")
        os.environ["MONTANDON_API_TOKEN"] = token
    return token


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_token()}"}


def connect(url: str = API_URL):
    """Open an authenticated :class:`pystac_client.Client` and report where it points."""
    from pystac_client import Client

    client = Client.open(url, headers=auth_headers())
    env = "production" if url.rstrip("/") == PRODUCTION_URL else (
        "staging" if url.rstrip("/") == STAGING_URL else "custom")
    print(f"Connected to {client.title!r} ({env}: {url})")
    return client


_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(auth_headers())
        _SESSION.headers["Accept"] = "application/geo+json"
    return _SESSION


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def _post(url: str, body: dict, retries: int = 4) -> dict:
    """POST with exponential back-off on transient errors (timeouts, 5xx, 429)."""
    for attempt in range(retries + 1):
        try:
            r = _session().post(url, json=body, timeout=300)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.Timeout, requests.ConnectionError) as exc:  # pragma: no cover - network
            err = repr(exc)
        if attempt == retries:
            raise RuntimeError(f"Search failed after {retries + 1} attempts - {err}")
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def search(
    collections: Sequence[str] | None = None,
    *,
    filter: dict | None = None,
    datetime: str | None = None,
    bbox: Sequence[float] | None = None,
    fields: Sequence[str] | None = None,
    sortby: Sequence[dict] | None = None,
    limit: int = 1000,
    max_items: int | None = None,
    url: str = API_URL,
    verbose: bool = True,
) -> list[dict]:
    """Run a STAC ``POST /search`` and follow every ``next`` link.

    Returns plain GeoJSON feature dictionaries (fast and memory friendly).
    Server-side filtering (``filter`` = CQL2-JSON) should always be preferred over
    downloading a whole collection and filtering in Python.

    ``fields`` uses the STAC *fields* extension to download only the properties you
    need, e.g. ``["id", "collection", "properties.datetime"]``. Geometry is dropped
    unless you include ``"geometry"``.

    ``max_items=None`` means *all* matching items: unlike ``pystac_client``'s
    ``max_items`` argument nothing is silently truncated. A warning is printed when
    ``max_items`` is reached so that totals are never mistaken for complete counts.
    """
    body: dict[str, Any] = {"limit": int(min(limit, max_items or limit))}
    if collections:
        body["collections"] = list(collections)
    if filter:
        body["filter"] = filter
        body["filter-lang"] = "cql2-json"
    if datetime:
        body["datetime"] = datetime
    if bbox:
        body["bbox"] = list(bbox)
    if fields:
        body["fields"] = {"include": list(fields)}
    if sortby:
        body["sortby"] = list(sortby)

    t0 = time.time()
    page = _post(f"{url}/search", body)
    items: list[dict] = list(page.get("features", []))
    while page.get("features"):
        if max_items and len(items) >= max_items:
            items = items[:max_items]
            print(f"  ! stopped at max_items={max_items}; more items match this query")
            break
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt:
            break
        if nxt.get("method", "GET").upper() == "POST":
            nbody = {**body, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body", body)
            page = _post(nxt["href"], nbody)
        else:
            r = _session().get(nxt["href"], timeout=300)
            r.raise_for_status()
            page = r.json()
        items.extend(page.get("features", []))
    if verbose:
        what = ", ".join(collections) if collections else "all collections"
        print(f"  {len(items):,} items from {what} ({time.time() - t0:.1f}s)")
    return items


# ---------------------------------------------------------------------------
# CQL2-JSON building blocks
# ---------------------------------------------------------------------------


def prop(name: str) -> dict:
    return {"property": name}


def cql_and(*args: dict | None) -> dict:
    args = [a for a in args if a]
    return args[0] if len(args) == 1 else {"op": "and", "args": list(args)}


def cql_or(*args: dict) -> dict:
    return {"op": "or", "args": list(args)}


def hazard_filter(codes: Iterable[str]) -> dict:
    """Items whose ``monty:hazard_codes`` array shares at least one code with *codes*."""
    return {"op": "a_overlaps", "args": [prop("monty:hazard_codes"), sorted(set(codes))]}


def country_filter(iso3: Iterable[str] | str) -> dict:
    codes = [iso3] if isinstance(iso3, str) else sorted(set(iso3))
    return {"op": "a_overlaps", "args": [prop("monty:country_codes"), codes]}


def time_filter(start: str, end: str) -> dict:
    """``datetime`` intersects the closed interval [start, end] (ISO-8601 strings)."""
    def iso(x: str) -> str:
        return x if "T" in x else (f"{x}T00:00:00Z" if x == start else f"{x}T23:59:59Z")
    return {"op": "t_intersects", "args": [prop("datetime"), {"interval": [iso(start), iso(end)]}]}


def eq(name: str, value: Any) -> dict:
    return {"op": "=", "args": [prop(name), value]}


def gte(name: str, value: Any) -> dict:
    return {"op": ">=", "args": [prop(name), value]}


def like(name: str, pattern: str) -> dict:
    return {"op": "like", "args": [prop(name), pattern]}


def is_in(name: str, values: Iterable[Any]) -> dict:
    return {"op": "in", "args": [prop(name), list(values)]}


# ---------------------------------------------------------------------------
# Hazard taxonomy: one family label across GLIDE, EM-DAT and UNDRR-ISC 2020/2025
# Source: https://ifrcgo.org/monty-stac-extension/model/taxonomy/
# ---------------------------------------------------------------------------

HAZARD_FAMILIES: dict[str, list[str]] = {
    "earthquake": ["EQ", "nat-geo-ear-gro", "GH0101", "GH0001", "GH0002", "GH0003", "GH0004"],
    "tsunami": ["TS", "nat-geo-ear-tsu", "MH0705", "MH0029", "GH0006"],
    "volcano": ["VO", "nat-geo-vol-", "GH0201", "GH0202", "GH0203", "GH0204", "GH0205"],
    "tropical_cyclone": ["TC", "nat-met-sto-tro", "MH0309", "MH0306", "MH0057", "MH0058", "MH0030"],
    "flood": ["FL", "FF", "nat-hyd-flo-", "nat-cli-glo-glo", "MH0600", "MH0601", "MH0602", "MH0603",
              "MH0604", "MH0605", "MH0606", "MH0607", "MH0608", "MH0609", "MH0610", "MH0004",
              "MH0005", "MH0006", "MH0007", "MH0008", "MH0009", "MH0010", "MH0011", "MH0012",
              "MH0013"],
    "landslide": ["LS", "nat-geo-mmd-", "nat-hyd-mmw-", "GH0300", "GH0301", "GH0302", "GH0303",
                  "GH0304", "GH0305", "GH0306", "GH0007", "GH0031", "GH0032"],
    "cold_wave": ["CW", "nat-met-ext-col", "nat-met-ext-sev", "nat-met-sto-bli", "MH0502", "MH0503",
                  "MH0504", "MH0505", "MH0403", "MH0405", "MH0406", "MH0040", "MH0041", "MH0042",
                  "MH0043", "MH0046", "MH0034", "MH0037", "MH0038", "MH0039"],
    "heat_wave": ["HT", "nat-met-ext-hea", "MH0501", "MH0047"],
    "drought": ["DR", "nat-cli-dro-dro", "MH0401", "MH0035"],
    "wildfire": ["WF", "nat-cli-wil-", "EN0205", "EN0013"],
    "storm": ["ST", "EC", "SS", "nat-met-sto-", "MH0307", "MH0308", "MH0102", "MH0103", "MH0404",
              "MH0703", "MH0031", "MH0032", "MH0099"],
}
# Order matters when an item carries codes from several families (e.g. a cyclone that
# also lists flooding): the first matching family wins.
FAMILY_PRIORITY = ["earthquake", "tsunami", "volcano", "tropical_cyclone", "flood", "landslide",
                   "cold_wave", "heat_wave", "drought", "wildfire", "storm"]


def codes_for(*families: str) -> list[str]:
    """All exact codes for the given families (prefix entries are expanded to known codes)."""
    known_prefix_expansions = {
        "nat-geo-vol-": ["nat-geo-vol-ash", "nat-geo-vol-lah", "nat-geo-vol-lav", "nat-geo-vol-pyr",
                         "nat-geo-vol-vol"],
        "nat-hyd-flo-": ["nat-hyd-flo-coa", "nat-hyd-flo-fla", "nat-hyd-flo-flo", "nat-hyd-flo-ice",
                         "nat-hyd-flo-riv"],
        "nat-geo-mmd-": ["nat-geo-mmd-ava", "nat-geo-mmd-lan", "nat-geo-mmd-roc", "nat-geo-mmd-sub"],
        "nat-hyd-mmw-": ["nat-hyd-mmw-ava", "nat-hyd-mmw-lan", "nat-hyd-mmw-mud", "nat-hyd-mmw-roc",
                         "nat-hyd-mmw-sub"],
        "nat-cli-wil-": ["nat-cli-wil-for", "nat-cli-wil-lan", "nat-cli-wil-wil"],
        "nat-met-sto-": ["nat-met-sto-der", "nat-met-sto-ext", "nat-met-sto-hai", "nat-met-sto-lig",
                         "nat-met-sto-san", "nat-met-sto-sev", "nat-met-sto-sto", "nat-met-sto-sur",
                         "nat-met-sto-tor"],
    }
    out: list[str] = []
    for fam in families:
        for code in HAZARD_FAMILIES[fam]:
            out.extend(known_prefix_expansions.get(code, [code]))
    return sorted(set(out))


def hazard_family(codes: Iterable[str] | None) -> str | None:
    """Map a ``monty:hazard_codes`` list to a single hazard family (or ``None``)."""
    codes = [c.strip() for c in (codes or []) if isinstance(c, str)]
    for fam in FAMILY_PRIORITY:
        for ref in HAZARD_FAMILIES[fam]:
            if any(c == ref or (ref.endswith("-") and c.startswith(ref)) for c in codes):
                # "storm" contains the generic nat-met-sto- prefix: don't let it swallow
                # tropical cyclones / blizzards, which are matched earlier.
                return fam
    return None


# ---------------------------------------------------------------------------
# Flattening items into tidy tables
# ---------------------------------------------------------------------------


def _representative_point(geom: dict | None) -> tuple[float, float]:
    if not geom or not geom.get("coordinates"):
        return (np.nan, np.nan)
    if geom["type"] == "Point":
        return tuple(geom["coordinates"][:2])  # type: ignore[return-value]
    try:
        from shapely.geometry import shape

        p = shape(geom).representative_point()
        return (p.x, p.y)
    except Exception:  # pragma: no cover - malformed geometry
        return (np.nan, np.nan)


def to_frame(items: Sequence[dict], geometry: bool = False) -> pd.DataFrame:
    """Flatten STAC feature dicts into a tidy :class:`pandas.DataFrame`.

    Columns: id, collection, source, role, datetime (UTC), start/end, title,
    country_codes (list), hazard_codes (list), hazard_family, corr_id, lon/lat,
    plus every ``monty:hazard_detail.*`` and ``monty:impact_detail.*`` field.
    Country codes are stripped of stray whitespace (a known source-data issue).
    """
    rows = []
    for f in items:
        p = f.get("properties", {}) or {}
        roles = p.get("roles") or []
        row = {
            "id": f.get("id"),
            "collection": f.get("collection"),
            "source": (f.get("collection") or "").rsplit("-", 1)[0],
            "role": next((r for r in ("event", "hazard", "impact", "response") if r in roles), None),
            "datetime": p.get("datetime"),
            "start_datetime": p.get("start_datetime"),
            "end_datetime": p.get("end_datetime"),
            "title": p.get("title"),
            "country_codes": [c.strip() for c in (p.get("monty:country_codes") or []) if c and c.strip()],
            "hazard_codes": p.get("monty:hazard_codes") or [],
            "corr_id": p.get("monty:corr_id"),
            "src_event_id": p.get("monty:src_event_id"),
        }
        for key in ("monty:hazard_detail", "monty:impact_detail", "monty:response_detail"):
            detail = p.get(key)
            if isinstance(detail, dict):
                prefix = key.split(":")[1].replace("_detail", "")
                for k, v in detail.items():
                    row[f"{prefix}_{k}"] = v
        for k in ("eq:magnitude", "eq:magnitude_type", "eq:depth", "eq:status"):
            if k in p:
                row[k.replace(":", "_")] = p[k]
        if f.get("geometry"):
            row["lon"], row["lat"] = _representative_point(f["geometry"])
            if geometry:
                row["geometry"] = f["geometry"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("datetime", "start_datetime", "end_datetime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["hazard_family"] = df["hazard_codes"].map(hazard_family)
    df["iso3"] = df["country_codes"].map(lambda c: c[0] if c else None)
    for col in ("impact_value", "hazard_severity_value", "eq_magnitude", "eq_depth"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # e.g. eq:depth == "-"
    return df


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, country) - needed when a record covers several countries."""
    out = df.explode("country_codes").rename(columns={"country_codes": "country"})
    return out[out["country"].notna() & (out["country"] != "UNK")]


# ---------------------------------------------------------------------------
# Impact bookkeeping: avoid double counting
# ---------------------------------------------------------------------------

EVENT_KEY_BY_SOURCE = {
    # Impact ids encode the source event, e.g. emdat-impact-2023-0054-TUR-total_deaths
    "emdat": r"^emdat-impact-(\d{4}-\d{4})",
    "gdacs": r"^gdacs-impact-(\d+)",
    "ifrcevent": r"^ifrcevent-impact-(\d+)",
    "usgs": r"^usgs-impact-([a-z0-9]+)-",
    "gfd": r"^gfd-impact-(\d+)",
}


def source_event_key(df: pd.DataFrame) -> pd.Series:
    """Best-effort source event identifier for impact rows (falls back to corr_id)."""
    key = pd.Series(index=df.index, dtype=object)
    for src, pattern in EVENT_KEY_BY_SOURCE.items():
        m = df["source"].eq(src)
        key[m] = df.loc[m, "id"].str.extract(pattern, expand=False)
    return key.fillna(df["src_event_id"] if "src_event_id" in df else None).fillna(df["corr_id"])


def latest_impacts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one value per (source, source event, country, impact type, unit).

    Sources such as IFRC and GDACS publish *successive updates* of the same figure
    (e.g. 35,418 → 36,187 → 40,642 deaths). Summing them inflates totals. We keep
    the maximum, which for cumulative humanitarian counts is the latest/most
    complete estimate.
    """
    d = df.copy()
    d["event_key"] = source_event_key(d)
    d["country"] = d["iso3"]
    keys = ["source", "event_key", "country", "impact_category", "impact_type", "impact_unit"]
    idx = d.groupby(keys, dropna=False)["impact_value"].idxmax().dropna().astype(int)
    return d.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-source event matching (corr_id is NOT a universal join key)
# ---------------------------------------------------------------------------


def match_events(
    df: pd.DataFrame,
    window_days: float = 7,
    max_km: float | None = None,
) -> pd.Series:
    """Cluster records from different sources that describe the same disaster.

    Two records are linked when they share a hazard family, have at least one
    country in common (or both lack a country but are within ``max_km``), and their
    dates are within ``window_days``. Linked records are merged transitively
    (union-find). Returns an integer ``event_uid`` per row.

    ``monty:corr_id`` encodes date, country, a spatial block and hazard code, so the
    same disaster often receives *different* corr_ids in different sources (e.g. the
    6 Feb 2023 Türkiye earthquake has >10). Exact corr_id joins therefore miss many
    true matches; this function is a transparent, tunable alternative.
    """
    d = df.reset_index(drop=True)
    parent = list(range(len(d)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # exact corr_id matches are always merged
    for _, idx in d.groupby("corr_id").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)

    t = d["datetime"].astype("int64").to_numpy() / 86400e9
    window = float(window_days)
    for fam, g in d.groupby("hazard_family"):
        g = g.assign(_t=t[g.index]).sort_values("_t")
        idx, times = g.index.to_numpy(), g["_t"].to_numpy()
        ctry = g["country_codes"].map(lambda c: set(c) - {"UNK"}).to_numpy()
        lon, lat = g.get("lon", pd.Series(np.nan, g.index)).to_numpy(), g.get("lat", pd.Series(np.nan, g.index)).to_numpy()
        start = 0
        for k in range(len(idx)):
            while times[k] - times[start] > window:
                start += 1
            for m in range(start, k):
                if ctry[k] & ctry[m]:
                    union(idx[k], idx[m])
                elif max_km and not ctry[k] and not ctry[m] and np.isfinite(lat[k]) and np.isfinite(lat[m]):
                    if haversine_km(lon[k], lat[k], lon[m], lat[m]) <= max_km:
                        union(idx[k], idx[m])
    roots = pd.Series([find(i) for i in range(len(d))])
    return pd.Series(pd.factorize(roots)[0], index=df.index, name="event_uid")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# External reference data (cached on disk)
# ---------------------------------------------------------------------------


def _cached_json(name: str, fetch, max_age_days: float = 30):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data


def world_bank(indicator: str, countries: Iterable[str] | str = "all", start: int = 1990,
               end: int = 2024) -> pd.DataFrame:
    """Fetch a World Bank WDI indicator (e.g. ``SP.POP.TOTL``) as tidy iso3/year/value rows."""
    ctry = countries if isinstance(countries, str) else ";".join(sorted(set(countries)))

    def fetch():
        url = f"https://api.worldbank.org/v2/country/{ctry}/indicator/{indicator}"
        out, page = [], 1
        while True:
            r = requests.get(url, params={"format": "json", "date": f"{start}:{end}", "per_page": 20000,
                                          "page": page}, timeout=180)
            r.raise_for_status()
            meta, rows = r.json()
            out += rows or []
            if page >= meta.get("pages", 1):
                return out
            page += 1

    key = f"wb_{indicator}_{abs(hash(ctry)) % 10**8}_{start}_{end}.json"
    rows = _cached_json(key, fetch)
    df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
                       for r in rows if r.get("countryiso3code")])
    return df.dropna(subset=["value"])


def noaa_index(name: str = "oni") -> pd.Series:
    """Monthly climate index from NOAA PSL as a Series indexed by month-start date.

    ``name``: ``"oni"`` (ENSO, Oceanic Niño Index) or ``"dmi"`` (Indian Ocean Dipole).
    """
    urls = {"oni": "https://psl.noaa.gov/data/correlation/oni.data",
            "dmi": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"}
    text = _cached_json(f"noaa_{name}.json", lambda: requests.get(urls[name], timeout=60).text, 7)
    lines = text.strip().splitlines()
    y0, y1 = map(int, lines[0].split()[:2])
    vals = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 13 and parts[0].isdigit() and y0 <= int(parts[0]) <= y1:
            for m, v in enumerate(parts[1:], start=1):
                v = float(v)
                if v > -99:
                    vals[pd.Timestamp(int(parts[0]), m, 1)] = v
    return pd.Series(vals, name=name.upper()).sort_index()


def country_boundaries():
    """Natural Earth 1:110m country polygons as a GeoDataFrame (iso3 in column ``iso3``)."""
    import geopandas as gpd

    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
           "ne_110m_admin_0_countries.geojson")
    gj = _cached_json("ne_110m_admin_0_countries.geojson", lambda: requests.get(url, timeout=120).json(), 365)
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    gdf["iso3"] = gdf["ISO_A3_EH"].where(gdf["ISO_A3_EH"] != "-99", gdf["ADM0_A3"])
    return gdf[["iso3", "NAME", "CONTINENT", "REGION_UN", "SUBREGION", "POP_EST", "geometry"]]


def describe_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Quick coverage table: items, first/last date per collection."""
    return (df.groupby("collection")
              .agg(items=("id", "size"), first=("datetime", "min"), last=("datetime", "max"))
              .sort_values("items", ascending=False))

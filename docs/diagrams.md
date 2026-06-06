# Repository Diagrams

These diagrams describe the repository structure and implemented pipeline. They
omit open research ideas and downstream analysis steps that are not part of the
artifact pipeline.

## High-Level Architecture

```mermaid
flowchart LR
    Scripts["scripts/ stage entry points"]
    Package["src/georeset_osm_web_evidence"]
    Data["data/ local artifacts"]
    Docs["docs/ and README"]
    Config["pyproject.toml, uv.lock, Dockerfile"]

    Scripts --> Package
    Package --> Data
    Docs --> Scripts
    Config --> Scripts
```

The scripts are thin command-line entry points. Most reusable behavior lives in
the Python package under `src/georeset_osm_web_evidence`.

## Module Map

```mermaid
flowchart TD
    Package["georeset_osm_web_evidence"]
    OSM["osm: Overpass queries, geometry, sampling"]
    Wikipedia["wikipedia: geosearch and point-in-polygon checks"]
    Search["search: query generation, Brave provider, coverage"]
    Web["web: page fetching and text extraction"]
    Review["review: human-review tables and XLSX export"]
    Storage["storage: GeoDataFrame parquet IO"]
    Viz["viz: Folium map generation"]

    Package --> OSM
    Package --> Wikipedia
    Package --> Search
    Package --> Web
    Package --> Review
    Package --> Storage
    Package --> Viz
```

## Data-Flow Pipeline

```mermaid
flowchart LR
    BBox["Configured France bboxes"]
    Overpass["Overpass OSM fetch"]
    RawOSM["data/raw/osm/named_polygon_candidates.parquet"]
    Wiki["Wikipedia coordinate geosearch"]
    WikiArtifact["data/interim/wikipedia/named_polygon_candidates_wikipedia.parquet"]
    Sample["Balanced sample"]
    SampleArtifact["data/processed/samples/balanced_wikipedia_100.parquet"]
    Brave["Brave Search results"]
    Results["data/processed/search/brave_results_sample.parquet"]
    URLs["data/processed/search/brave_candidate_urls_sample.parquet"]
    Text["Fetched page text"]
    Evidence["data/processed/evidence/page_text_sample.parquet"]
    Review["Human review CSV/XLSX"]
    ReviewArtifact["data/review/human_review_sample.*"]

    BBox --> Overpass --> RawOSM
    RawOSM --> Wiki --> WikiArtifact
    WikiArtifact --> Sample --> SampleArtifact
    SampleArtifact --> Brave --> Results
    Results --> URLs
    URLs --> Text --> Evidence
    Evidence --> Review --> ReviewArtifact
```

The search URL stage removes Wikipedia URLs before page-text fetching. This is a
known limitation because coordinate-based Wikipedia geosearch can miss relevant
articles without coordinates.

## Script Workflow

```mermaid
flowchart TD
    FetchOSM["scripts/osm/fetch_osm_polygons.py"]
    WikiBatch["scripts/wikipedia/check_wikipedia_geosearch_batch.py"]
    Sample["scripts/osm/sample_balanced_wikipedia_polygons.py"]
    Map["scripts/osm/visualize_osm_samples.py"]
    QueryBuild["scripts/search/build_search_queries.py"]
    Coverage["scripts/search/report_brave_search_coverage.py"]
    BraveCollect["scripts/search/collect_brave_search_results.py"]
    URLPrep["scripts/search/prepare_search_result_urls.py"]
    FetchText["scripts/evidence/fetch_candidate_page_text.py"]
    ReviewBuild["scripts/review/build_human_review_sample.py"]

    FetchOSM --> WikiBatch --> Sample
    Sample --> Map
    Sample --> QueryBuild
    Sample --> Coverage
    Coverage --> BraveCollect --> URLPrep --> FetchText --> ReviewBuild
```

`search/search_brave_sample.py` is a small Brave API smoke test and is excluded
from the main workflow because it is not part of the artifact pipeline.

## Artifact Lifecycle

```mermaid
flowchart TD
    Raw["raw: OSM candidate polygons"]
    Interim["interim: Wikipedia-enriched candidates"]
    Processed["processed: samples, search, evidence, maps"]
    Review["review: human annotation files"]
    HF["Hugging Face bucket sync"]

    Raw --> Interim --> Processed --> Review
    Raw -. optional sync .-> HF
    Interim -. optional sync .-> HF
    Processed -. optional sync .-> HF
    Review -. optional sync .-> HF
```

The repository keeps code in Git and treats `data/` as generated local artifacts.
The project README points to the Hugging Face bucket used for external data
storage.

## Config And Runtime

```mermaid
flowchart LR
    PyProject["pyproject.toml: package metadata and dependencies"]
    Lock["uv.lock: locked dependency graph"]
    Docker["Dockerfile: uv-based Python image"]
    Citation["CITATION.cff: citation metadata"]
    Ignore[".gitignore and .dockerignore"]

    PyProject --> Lock
    Lock --> Docker
    Ignore --> Docker
    Citation --> Docs["Repository metadata"]
```

The runtime is Python 3.10+ with `uv`; Brave Search requires
`BRAVE_SEARCH_API_KEY`.

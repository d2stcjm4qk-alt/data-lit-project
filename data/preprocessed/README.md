# OSM Highway Filter

Filter major roads from OpenStreetMap (OSM) using **Osmium Tool** as raw files are too large to be fully loaded.

## Setup

```bash
conda create -n osmium_env python=3.11
conda activate osmium_env
conda install -c conda-forge osmium-tool

## Usage
osmium tags-filter "path/to/input.osm.pbf" \
  w/highway=motorway w/highway=motorway_link \
  w/highway=trunk w/highway=trunk_link \
  w/highway=primary w/highway=primary_link \
  -o "path/to/output.osm.pbf" --overwrite

Apply this for both, UK and Germany streetmaps and only work with these!

# Data Directory

This directory contains all datasets used in the analysis, separated into **raw** and **preprocessed** data.

- `raw/` contains unprocessed datasets as obtained from the original sources (e.g., OpenStreetMap and national accident statistics).
- `preprocess/` contains filtered and cleaned datasets used directly in the analysis.

Because several datasets exceed GitHub’s file size limits, a link to the **complete data archive** (including all raw and processed files) is provided [here](https://unitc-my.sharepoint.com/:f:/g/personal/zxosr65_s-cloud_uni-tuebingen_de/IgDBnJlNNBvpRIr6TGh6QUZNAfX6Ivh91Fe16lmBpICeN8w?e=qmEOZu).

For OpenStreetMap (OSM) data, raw `.osm.pbf` files are very large. We therefore strongly recommend filtering major road classes using **Osmium Tool** before further processing. All analyses are based on these filtered OSM datasets for **Germany** and the **UK**.

## OSM Highway Filter

Filter major roads from OpenStreetMap (OSM) using **Osmium Tool** as raw files are too large to be fully loaded.

### Setup

```bash
conda create -n osmium_env python=3.11
conda activate osmium_env
conda install -c conda-forge osmium-tool

### Usage
osmium tags-filter "path/to/input.osm.pbf" \
  w/highway=motorway w/highway=motorway_link \
  w/highway=trunk w/highway=trunk_link \
  w/highway=primary w/highway=primary_link \
  -o "path/to/output.osm.pbf" --overwrite

Apply this for both, UK and Germany streetmaps and only work with these!

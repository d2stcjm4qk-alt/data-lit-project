# Data Directory

This directory contains all datasets used in the analysis, separated into **raw** and **preprocessed** data.

- `raw/` contains unprocessed datasets as obtained from the original sources (e.g., OpenStreetMap and national accident statistics).
- `preprocess/` contains filtered and cleaned datasets used directly in the analysis.

Because several datasets exceed GitHub’s file size limits, the **complete data archive**, including all raw and processed files, is provided directly [here](https://unitc-my.sharepoint.com/:f:/g/personal/zxosr65_s-cloud_uni-tuebingen_de/IgDBnJlNNBvpRIr6TGh6QUZNAfX6Ivh91Fe16lmBpICeN8w?e=qmEOZu). We strongly recommend using this bundled data folder, as reproducing the dataset would otherwise require collecting multiple sources from different providers.

For OpenStreetMap (OSM) data, raw `.osm.pbf` files are very large. We therefore strongly recommend filtering major road classes using **Osmium Tool** before further processing. All analyses are based on these filtered OSM datasets for **Germany** and the **UK**.

## Links to all datasets

### Accidents
- Germany: https://www.opengeodata.nrw.de/produkte/transport_verkehr/unfallatlas/Unfallorte2024_EPSG25832_CSV.zip
- UK: https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv, https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-2024.csv, https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-2024.csv

### Population
- UK: https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2024/mye24tablesuk.xlsx
- Germany: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/04-kreise.html
- Berlin: https://download.statistik-berlin-brandenburg.de/eac90c8b858f8cac/9d46fc1fd059/SB_A01-05-00_2025h01_BE.xlsx
- Munich: https://opendata.muenchen.de/dataset/e3f5dbd2-39cc-40cd-bc91-4bb49a0b1802/resource/a641ce6a-4e01-4f4b-9976-1ae6a47e3762/download/bevolkerung_bezirke_neu.csv
- Hamburg: https://hub.arcgis.com/datasets/esri-de-content::stadtteile-hamburg/explore?location=53.567224%2C10.027704%2C11

### Regions
- UK: https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-may-2024-boundaries-uk-bgc-2/about
- Germany: https://daten.gdz.bkg.bund.de/produkte/vg/vg250-ew_ebenen_1231/aktuell/vg250-ew_12-31.utm32s.gpkg.ebenen.zip, https://daten.gdz.bkg.bund.de/produkte/vg/vg250-ew_ebenen_1231/aktuell/vg250-ew_12-31.utm32s.shape.ebenen.zip
- Berlin: https://daten.berlin.de/datensaetze/rbs-bezirke-dezember-2015
- Munich: https://hub.arcgis.com/datasets/f7f4d6e8090742c6b774379780bd7d9b_0/explore?location=48.154871%2C11.541844%2C11
- Hamburg: https://hub.arcgis.com/datasets/esri-de-content::stadtteile-hamburg/explore?location=53.567224%2C10.027704%2C11

### Traffic
- Germany: https://www.bast.de/DE/Themen/Digitales/HF_1/Massnahmen/verkehrszaehlung/Daten/2023_1/Jawe2023.html?filter=true&nn=401522&map=0
- UK: https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip

### OSM
- Germany: https://download.geofabrik.de/europe/germany.html
- UK: https://download.geofabrik.de/europe/united-kingdom-251201.osm.pbf

### OSM Highway Filter

Filter major roads from OpenStreetMap (OSM) using **Osmium Tool** as raw files are too large to be fully loaded.

#### Setup

```bash
conda create -n osmium_env python=3.11
conda activate osmium_env
conda install -c conda-forge osmium-tool

#### Usage
osmium tags-filter "path/to/input.osm.pbf" \
  w/highway=motorway w/highway=motorway_link \
  w/highway=trunk w/highway=trunk_link \
  w/highway=primary w/highway=primary_link \
  -o "path/to/output.osm.pbf" --overwrite
```
Apply this for both, UK and Germany streetmaps and only work with these!

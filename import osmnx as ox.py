import osmnx as ox
import matplotlib.pyplot as plt

# Logging optional
ox.settings.log_console = True
ox.settings.use_cache = True

# Filter nur für Autobahnen
motorway_filter = (
    '["highway"~"motorway|motorway_link"]'
)

# Deutschland-BoundingBox (ungefähr)
north = 55.1
south = 47.2
east  = 15.0
west  = 5.9

# Straßennetz laden
G = ox.graph_from_bbox(
    north, south, east, west,
    custom_filter=motorway_filter
)

# Plotten
fig, ax = ox.plot_graph(
    G,
    node_size=0,
    edge_color="blue",
    edge_linewidth=0.6,
    bgcolor="white"
)
"""Real-world maps: turning OpenStreetMap data into a SENTRY disaster city.

* :mod:`~sentry_ai.mapping.osm` — pure: OSM ways in, a map config dict out
  (the same schema ``configs/maps/city_default.yaml`` uses), with the
  disaster — victims, fires, collapsed buildings — placed so every victim
  is reachable.
* :mod:`~sentry_ai.mapping.overpass` — the network side: geocoding a place
  name and fetching its streets and buildings, with a local cache.

The import runs once, ahead of time; the generated map file is what a
mission loads, so a demo never needs a network connection.
"""

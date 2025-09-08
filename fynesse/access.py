#Complete wrt 01-geospatial-practical
from .config import *
import osmnx as ox
import matplotlib.pyplot as plt
import math

def get_bbox(latitude,longitude,box_size_km):
    '''
    Takes location and box size in km
    Returns bounding box coordinates (square)

    '''
    # Convert km to degrees
    lat_offset = (box_size_km / 2) / 111
    lon_offset = (box_size_km / 2) / (111 * math.cos(math.radians(latitude)))


    north = latitude + lat_offset
    south = latitude - lat_offset
    east = longitude + lon_offset
    west = longitude - lon_offset


    return west, south, east, north

def get_osm_datapoints(latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Function for getting OSM data

    Parameters
    ----------
    place_name : str
        Name of the place (used for boundary + plot title).
    latitude, longitude : float
        Central coordinates.
    box_size_km : float
        Size of the bounding box in kilometers (default 2 km).
    poi_tags : dict, optional
        Tags dict for POIs (e.g. {"amenity": ["school", "restaurant"]}).
    
    Returns
    -------
    bbox : tuple
        Bounding box (west, south, east, north).
    nodes : GeoDataFrame
        OSM nodes.
    edges : GeoDataFrame
        OSM edges.
    buildings : GeoDataFrame
        OSM buildings.
    pois : GeoDataFrame or None
        OSM points of interest (if poi_tags provided), else None.
    """
    
    bbox = get_bbox(latitude,longitude,box_size_km)

    # Road graph
    graph = ox.graph_from_bbox(bbox, network_type="all")
    nodes, edges = ox.graph_to_gdfs(graph)

    # Buildings & POIs
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    pois = None
    if poi_tags:
        pois = ox.features_from_bbox(bbox, tags=poi_tags)
    
    # Ensure correct geometry column
    nodes = nodes.set_geometry("geometry")
    edges = edges.set_geometry("geometry")
    buildings = buildings.set_geometry("geometry")
    
    if pois is not None:
        pois = pois.set_geometry("geometry")
    
    return bbox, nodes, edges, buildings, pois

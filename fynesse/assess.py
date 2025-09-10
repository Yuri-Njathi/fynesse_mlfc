from typing import Any, Union, Tuple, Dict, Optional, cast
import matplotlib.pyplot as plt
import math
import osmnx as ox
import pandas as pd
import logging
import yaml
import os

from .config import *
from . import access

from .access import * 
# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def get_box(
    latitude: float, longitude: float, box_size_km: Union[float, int] = 2
) -> Tuple[float, float, float, float]:
    """
    args:
        latitude in degrees
        longitude in degrees

    returns:
        bounding box coordinates (west, south, east, north) in degrees
    """
    lat_degree_km = 111.0
    lon_degree_km = 111.0 * math.cos(math.radians(latitude))
    box_height_deg = box_size_km / lat_degree_km
    box_width_deg = box_size_km / lon_degree_km
    north = latitude + box_height_deg / 2
    south = latitude - box_height_deg / 2
    east = longitude + box_width_deg / 2
    west = longitude - box_width_deg / 2
    bbox = (west, south, east, north)
    return bbox


def load_default_tags() -> Dict[str, Union[bool, str, list[str]]]:
    # Open the Defaults YAML file and load it
    defaults_file_path = os.path.join(os.path.dirname(__file__), "defaults.yml")
    print(defaults_file_path)
    with open(defaults_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Access the DEFAULT_TAGS
    default_tags = cast(Dict[str, Union[bool, str, list[str]]], config["DEFAULT_TAGS"])
    return default_tags


def plot_city_map(
    place_name: str,
    latitude: float,
    longitude: float,
    box_size_km: Union[float, int] = 2,
    poi_tags: Optional[Dict[str, Union[bool, str, list[str]]]] = load_default_tags(), #None,
) -> None:
    # def plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=None):
    '''
    Access and visualize geographic data
    '''
    #generate bounding box
    bbox = get_bbox(latitude,longitude,box_size_km)
    west, south, east, north = bbox
    # POIs
    pois = get_pois(bbox,poi_tags) 
    # Get graph from location
    graph = ox.graph_from_bbox(bbox)#,# network_type="all")
    # City area
    #area = ox.geocode_to_gdf(place_name)
    # Street network
    nodes, edges = ox.graph_to_gdfs(graph)
    # Buildings
    buildings = ox.features_from_bbox(bbox, tags={"building": True})

    fig, ax = plt.subplots(figsize=(6,6))
    #area.plot(ax=ax, color="tan", alpha=0.5)
    buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    pois.plot(ax=ax, color="green", markersize=5, alpha=1)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    plt.show()

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Given a central point (latitude, longitude) and a bounding box size,
    query OpenStreetMap via OSMnx and return a feature vector.

    Parameters
    ----------
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size : float
        Size of the bounding box in kilometers
    features : list of tuples
        List of (key, value) pairs to count. Example:
        [
            ("amenity", None),
            ("amenity", "school"),
            ("shop", None),
            ("tourism", "hotel"),
        ]

    Returns
    -------
    feature_vector : dict
        Dictionary of feature counts, keyed by (key, value).
    """
    bbox = get_bbox(latitude,longitude,box_size_km)
    west, south, east, north = bbox
    # POIs
    pois = get_pois(bbox,poi_tags) 
    # # Construct bbox from lat/lon and box_size
    # box_width = box_size_km*0.1/11 # About 11 km
    # box_height = box_size_km*0.1/11
    # north = latitude + box_height/2
    # south = latitude - box_height/2
    # west = longitude - box_width/2
    # east = longitude + box_height/2
    # bbox = (west, south, east, north)
    # # Query OSMnx for features
    # try:
    #   pois = ox.features_from_bbox(bbox, tags)
    # except Exception as e:
    #   print(f"Error querying OSMnx: {e}")
    #   return {}
    # if len(pois) == 0:
    #     return {}
    
    # print(len(pois)
    #print(pois.head())
    #print('--------------------------------------------------')
    # Count features matching each (key, value) in poi_types
    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)

    #print(pois_df.head())
    #print('--------------------------------------------------')
    #tourist_places_df = pois_df[pois_df.tourism.notnull()]
    # Return dictionary of counts
    poi_counts = {}
    for key, value in features:
        if key in pois_df.columns:
            if value:  # count only that value
                poi_counts[f"{key}:{value}"] = (pois_df[key] == value).sum()
            else:  # count any non-null entry
                poi_counts[key] = pois_df[key].notnull().sum()
        else:
            poi_counts[f"{key}:{value}" if value else key] = 0
    # poi_counts_df = pd.DataFrame(list(poi_counts.items()), columns=["POI Type", "Count"])
    # poi_counts_df # feature vector/poi_counts

    #raise NotImplementedError("Feature extraction not implemented yet.")
    return poi_counts

#plotting func for dsail porini
def custom_poi_plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=load_default_tags(), custom_poi=None):
    '''
    Access and visualize geographic data
    '''
    bbox = get_bbox(latitude, longitude, box_size_km)
    west, south, east, north = bbox

    pois = get_pois(bbox, poi_tags)
    graph = ox.graph_from_bbox(bbox)
    area = ox.geocode_to_gdf(place_name)
    nodes, edges = ox.graph_to_gdfs(graph)
    buildings = ox.features_from_bbox(bbox, poi_tags)

    fig, ax = plt.subplots(figsize=(6, 6))
    buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    pois.plot(ax=ax, color="green", markersize=5, alpha=1)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)

    # Different colors for each cluster
    colors = ["red", "blue", "purple", "orange", "cyan"]

    for i, coords in enumerate(custom_poi):
        lats = [v[0] for v in coords.values()]
        lons = [v[1] for v in coords.values()]
        ax.scatter(lons, lats, c=colors[i % len(colors)], s=40, marker="o", label=f"Cluster {i}")

        # Add labels
        for key, (lat, lon) in coords.items():
            ax.text(lon, lat, key, fontsize=8, ha="right", va="bottom")

    ax.legend()
    plt.show()

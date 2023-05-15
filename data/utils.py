import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon
import numpy as np
import random

import requests
import os
import gzip
import zipfile
import shutil


HIGHWAY_DIST_THRESH = {
    "motorway": 15,
    "trunk": 15,

    "motorway_link": 13,
    "trunk_link": 13,
    "primary": 13,

    "primary_link": 11,

    "secondary": 7,

    "secondary_link": 5,
    "tertiary": 5,

    "tertiary_link": 4
}

TRANSPORT_VELOCITY_THRESHS = [
    # adapted from Dabiri et al., 2020, but less restrictive
    ("walk",       10),
    ("run",        15),
    ("bike",       20),
    ("bus",        40),
    ("train",      60),
    ("subway",     60),
    ("car",        60),
    ("taxi",       60),
    ("motorcycle", 60),
    ("boat",       60),
    ("airplane",  400),
]

GPS_CRS = "EPSG:4326"



def download_dataset(url:str, folderpath:str = "tmp", output_filename=None):
    filename = url.rsplit('/', 1)[-1]
    filepath = os.path.join(folderpath, filename)

    root, ext = os.path.splitext(filename)
    zipped = ext in [".gz", ".zip"]
    if output_filename is None:
        if zipped:
            output_filename = root
        else:
            output_filename = filename
    output_path = os.path.join(folderpath, output_filename)

    if not (os.path.exists(filepath) or os.path.exists(output_path)):
        # start download
        r = requests.get(url)
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        if zipped:
            downloaded_filepath = filepath
        else:
            downloaded_filepath = output_path

        with open(downloaded_filepath,'wb') as f:
            f.write(r.content)

        # unzip files
        if ext == ".gz":
            #not tested
            with gzip.open(downloaded_filepath, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(downloaded_filepath)

        elif ext == ".zip":
            with zipfile.ZipFile(downloaded_filepath, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            os.remove(downloaded_filepath)
    
    return output_path


def groupby_val(d:dict) -> dict:
    """group dict by val"""
    res = {}
    for i, v in d.items():
        res[v] = [i] if v not in res.keys() else res[v] + [i]
    return res


def remove_outliers(gdf:gpd.GeoDataFrame, def_vel_treshhold:float=400)->pd.DataFrame:
    """
    Remove location outliers based on max velocity threshold (default 400 m/s).
    Depending on transportation mode (Dabiri & Heaslip, 2018)
    """
    assert not gdf.crs.utm_zone is None, "CRS is not in UTM format"
    len_before_iter = np.inf

    # while still outliers detected
    while(len(gdf) < len_before_iter):
        gdf = gdf.reset_index(drop=True)
        len_before_iter = len(gdf)
        # add additional columns
        gdf["time_diff"] = (gdf.time - gdf.time.shift(1)).dt.total_seconds()
        gdf["distance"] = gdf.geometry.distance(gdf.geometry.shift(1))
        gdf["velocity"] = gdf["distance"]/gdf.time_diff
        gdf.loc[0, ["time_diff", "distance", "velocity"]] = 0
        if "transport" in gdf:
            for transport, vel_treshhold in TRANSPORT_VELOCITY_THRESHS:
                gdf = gdf[~(
                    (gdf.transport == transport) &
                    (gdf.velocity > vel_treshhold) &
                    (gdf.velocity.shift(-1) > vel_treshhold)
                )]
            gdf = gdf[~(
                gdf.transport.isna() &
                (gdf.velocity > def_vel_treshhold) &
                (gdf.velocity.shift(-1) > def_vel_treshhold)
            )]
        else:
            gdf = gdf[~(
                (gdf.velocity > def_vel_treshhold) &
                (gdf.velocity.shift(-1) > def_vel_treshhold)
            )]
    
    return gdf


def get_track_id(gdf:gpd.GeoDataFrame, time_thresh_split_sec:float=20*60,  def_vel_treshhold:float=400):
    """
    Splits dataset into tracks by user, time and location. Velocity works here
    as jump detection between trajectory

    time_thresh_split_sec: Threshold to split trajectories in seconds, Moreau 
    et al. (2021) and others used 20min
    """
    track_change = gdf.user != gdf.user.shift(1)
    track_change = track_change | (gdf.time_diff > time_thresh_split_sec)
    if "transport" in gdf:
        for transport, vel_treshhold in TRANSPORT_VELOCITY_THRESHS:
            track_change = track_change | ((gdf.transport == transport) & (gdf.velocity > vel_treshhold))
        track_change = track_change | (gdf.transport.isna() & (gdf.velocity > def_vel_treshhold))
    else:
        track_change = track_change | (gdf.velocity > def_vel_treshhold)
    return track_change.cumsum()


def c_by_amenity(gdf:gpd.GeoDataFrame, amenities_mean_area:float, osm_amenities:pd.DataFrame=None)-> pd.Series:
    """confidence for a stay point by amenity data from OSM"""
    if not osm_amenities is None:
        amenity_traj = gdf.amenity.explode()
        area_traj =  pd.merge(amenity_traj, osm_amenities[["amenity","area"]],
            how="left", on="amenity").set_index(amenity_traj.index).area
        weak_c = area_traj.apply(lambda area: np.exp(-area/amenities_mean_area))
        return weak_c.groupby(area_traj.index).max().fillna(0)
    else:
        return pd.Series(0, index=gdf.index)


def weak_label(gdf:gpd.GeoDataFrame, amenities_mean_area:float, osm_amenities:pd.DataFrame=None) -> pd.Series:
    """
    Label function that estimates a probability `p_final` for each point being a stay point and
    a confidence score `c_total` that indicates the confidence in the prediction (0 -> no confidence,
    but can exceed 1)
    """
    if len(gdf) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    c_amenity = c_by_amenity(gdf, amenities_mean_area, osm_amenities)

    c_close_to_highway = gdf.highway.notna().astype(int)
    c_inside_building = gdf.osmid.notna().astype(int)
    if 'transport' in gdf:
        c_transport = (gdf.transport.notna() & (~gdf.transport.isin(["walk", "run", "bike"]))).astype(int)
    else:
        c_transport = pd.Series(0, index=gdf.index)

    c_stay = c_amenity + c_inside_building
    c_non_stay = c_close_to_highway + c_transport
    c_total = c_stay + c_non_stay

    p_final = pd.Series(0.5, index=gdf.index) #0.5 is a placeholder value and is not used/masked in calculations
    p_final[c_total>0] = c_stay / c_total

    return p_final, c_total


def kfold_split_by_users(gdf:gpd.GeoDataFrame, label_weight_fun,
    k:int=5) -> list[tuple[gpd.GeoDataFrame]]:
    """
    Split dataset into k even large data sets according to users.
    At the end each user will appear only in one of the two sets,
    which prevents leakage (see Moreau et al. (2021))
    """
    random.seed(0, version=2)
    labels_per_user = label_weight_fun(gdf).groupby(by=gdf.user).sum()
    subsets = [labels_per_user.index.tolist()] + [[] for _ in range(k-1)]

    labels_per_subset = [labels_per_user[subset].sum() for subset in subsets]
    diff_percentage = (max(labels_per_subset) - min(labels_per_subset))/sum(labels_per_subset)
    counter = 0
    while diff_percentage > 0.01 and counter < 1E4: # tolerance
        max_set_idx = np.argmax(labels_per_subset)
        min_set_idx = np.argmin(labels_per_subset)
        user_id = random.choice(subsets[max_set_idx])
        subsets[max_set_idx].remove(user_id) # take from biggest set
        subsets[min_set_idx].append(user_id) # and put it to smallest set until they are almost even
        labels_per_subset = [labels_per_user[subset].sum() for subset in subsets]
        diff_percentage = (max(labels_per_subset) - min(labels_per_subset))\
            /sum(labels_per_subset)
        counter += 1
    
    splits = []
    for idx in range(k):
        train_set = sum(subsets[:idx] + subsets[idx+1:], [])
        test_set = subsets[idx]

        splits.append(tuple([
            gdf[gdf.user.isin(subset)].reset_index(drop=True)
            for subset in [train_set, test_set]
        ]))


    return splits


def enrich_with_close_highways(gdf:gpd.GeoDataFrame, highways:gpd.GeoDataFrame) -> pd.Series:
    """
        returns bool Series, if gdf points are on highway or not
        (determined by an intersection of highway and rectangulars
        around each point)
        gdf: trajectory
        sn: street network
        both needs to be in the same UTM zone
    """
    assert not gdf.crs.utm_zone is None, "CRS is not in UTM format"
    assert gdf.crs.utm_zone == highways.crs.utm_zone,\
        f"crs is not equal.\ngdf.crs={gdf.crs}\nhighways.crs={highways.crs}"

    visited_highway = pd.Series(None, index=gdf.index, dtype=float)

    for rect_margin, highway_types in groupby_val(HIGHWAY_DIST_THRESH).items():
        focused_hw_types = highways[highways.highway.isin(highway_types)]
        if len(focused_hw_types) > 0:
            remaining = visited_highway.isna()
            rectangulars = gdf[remaining].geometry.apply(
                lambda p: Polygon([
                    (p.x-rect_margin, p.y-rect_margin),
                    (p.x-rect_margin, p.y+rect_margin),
                    (p.x+rect_margin, p.y+rect_margin),
                    (p.x+rect_margin, p.y-rect_margin)
                ])) # rectangular around point p
            visited_highway_dupl = gpd.sjoin(gpd.GeoDataFrame(geometry=rectangulars, crs=gdf.crs),
                focused_hw_types, how="left", predicate="intersects").osmid
            visited_highway[remaining] = visited_highway_dupl.groupby(visited_highway_dupl.index).first()
    return visited_highway


def enrich_with_osm_contex(trajectory: gpd.GeoDataFrame, rois: gpd.GeoDataFrame=None,
    highways: gpd.GeoDataFrame=None, y_min: float=None, y_max: float=None,
    x_east: float=None, x_west: float=None, verbose:bool=False):
    """
    Weak label the trajectory with the ROIs.
    """

    if rois is None or highways is None:
        # load ROIs from OSM
        y_min = y_min if y_min is not None else trajectory.geometry.y.min()
        y_max = y_max if y_max is not None else trajectory.geometry.y.max()

        assert (x_east is not None and x_west is not None) or (x_east == x_west), \
        "x_west and x_east need to be specificed together or not at all"
        if x_west is None: 
            x_max = trajectory.geometry.x.max()
            x_min = trajectory.geometry.x.min()
            if x_max - x_min < 180:
                x_east = x_max
                x_west = x_min
            else:  # if the trajectory is crossing the 180° meridian
                x_east = x_min
                x_west = x_max

    if rois is None:
        rois = ox.geometries.geometries_from_bbox(y_max, y_min, x_east, x_west,
            tags={"building": True, "amenity": True}).reset_index()
        if verbose: print("rois downloaded to cache folder")
        rois = rois[rois.element_type != "node"] # remove single point buildings
    
    try:
        if highways is None:
            street_network = ox.graph.graph_from_bbox(y_max, y_min, x_east, x_west,
                custom_filter=f'["highway"~"{"|".join(HIGHWAY_DIST_THRESH.keys())}"]',
                truncate_by_edge=True, retain_all=True)
            _, highways = ox.utils_graph.graph_to_gdfs(street_network)
            if verbose: print("highways downloaded to cache folder")
        
        highways = highways[["geometry", "highway", "osmid"]]
        # remove highways of multiple types/with multiples osmids,
        # since they require more handling

        highways = highways[highways["highway"].apply(type) != list]
        highways = highways[highways["osmid"].apply(type) != list]

        if verbose: print("looking for near highways...", end="")
        trajectory["highway"] = enrich_with_close_highways(ox.project_gdf(trajectory), ox.project_gdf(highways))
        if verbose: print(" finished.")
        visited_highways = highways[highways.osmid.isin(trajectory.highway.dropna().unique())]
        if verbose: print("attached highways to trajectory")
    except ValueError as err: #Found no graph nodes within the requested polygon
        print(err)
        trajectory["highway"] = None
        visited_highways = None
    
    if not "building" in rois:
        trajectory["amenity"] = None
        trajectory["osmid"] = None
        return None, None, visited_highways, trajectory

    buildings = rois[~rois.building.isna()]
    buildings = buildings[["geometry", "osmid"]]#TODO: .rename(columns={"osmid": "buildings"})

    # filter outer buildings like this: https://www.openstreetmap.org/relation/7940925
    # 'intersects' is actually more accurate here, but then it needs to be decided,
    # which of the intersecting buildings will be removed. So it is easier to go with 'within'
    # and remvoe the outer buildings. The rest of "only" intersecting building are removed to
    # avoid duplicate points
    join = gpd.sjoin(buildings, buildings, how="left", predicate="within")
    outer_buildings = join.query("osmid_left != osmid_right").osmid_right.unique()
    buildings = buildings[~buildings.osmid.isin(outer_buildings)]

    # if still two buildings are intersecting, remove just one of them
    join = gpd.sjoin(buildings, buildings, how="left", predicate="intersects")
    intersecting_building = join.query("osmid_left != osmid_right").osmid_right.unique()
    buildings = buildings[~buildings.osmid.isin(intersecting_building)]
    if verbose: print("filtered buildings")

    amenities = rois[rois.building.isna()]
    amenities = amenities[["geometry", "osmid"]].rename(columns={"osmid": "amenity"})
    if verbose: print("filtered amenities")
    
    # add building ids to df
    merge = gpd.sjoin(trajectory, buildings, how="left", predicate="within").drop(labels="index_right", axis=1)
    assert ~merge.index.duplicated().any(),\
        f"Location points \n{merge[merge.index.duplicated()]} found in two different ROIs"
    if verbose: print("attached buildings to trajectiory")
    buildings = buildings[buildings.osmid.isin(merge.osmid.dropna().unique())]
    if len(buildings)>0:
        buildings["area"] = ox.project_gdf(buildings).area
    else:
        buildings["area"] = pd.Series()

    # add list of amenity ids to df
    merge = gpd.sjoin(merge, amenities, how="left", predicate="within").drop(labels="index_right", axis=1)
    amenities = amenities[amenities.amenity.isin(merge.amenity.dropna().unique())]
    if len(amenities)>0:
        amenities["area"] = ox.project_gdf(amenities).area
    else:
        amenities["area"] = pd.Series()
    
    merge["amenity"] = merge.groupby(merge.index).agg({"amenity": lambda x: x.dropna().astype(int).tolist()})
    merge = merge[~merge.index.duplicated()]
    if verbose: print("attached amneities to trajectiory")

    return buildings, amenities, visited_highways, merge

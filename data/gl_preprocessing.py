import utils
import os
import pandas as pd
import geopandas as gpd
import datetime as dt
import numpy as np
from tqdm import tqdm

# constants


file_dir = os.path.dirname(os.path.abspath(__file__))

GL_PROJ = "EPSG:32650" #for euclidean calculations like distance or area
AMENITIES_MEAN_AREA = 43092.96 # mean area of all visited amenities in preprocessed GeoLife

# code

def download()->str:
    """Download Geolife dataset"""
    url = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife Trajectories 1.3.zip"
    return utils.download_dataset(url, os.path.join(file_dir, "tmp"))


def is_in_beijing_region(df:pd.DataFrame)->pd.DataFrame:
    """
    Filter data on Bejing region, by using the 5%/95% quantiles as thresholds
    """
    return (df.lon > 108.47) &\
           (df.lon < 117.97) &\
           (df.lat > 30.847) &\
           (df.lat < 40.372)

def is_inner_beijing(df:pd.DataFrame)->pd.DataFrame:
    """
    Filter data on Bejing city, by using the 15%/85% quantiles as thresholds
    """
    return (df.lon > 116.242649495) & \
        (df.lon < 116.476522) & \
        (df.lat > 39.72573404499) & \
        (df.lat < 40.030812)


def tm_unlabelled_mask_fun(track):
    "returns True if Transportation Mode is available per row"
    track["bike"] = track.transport == "bike"
    track["bus"] = track.transport == "bus"
    track["walk"] = track.transport == "walk"
    track["drive"] = track.transport.isin(["car", "taxi"])
    track["train_"] = track.transport.isin(["train", "subway"])
    return track[["bike", "bus", "walk", "drive", "train_"]].sum(1) > 0


def add_transportation_mode_col(df:pd.DataFrame, data_folder)->pd.DataFrame:
    """
    Add transportation mode column, if information is available, else
    the old dataframe is returned again
    """
    df = df.reset_index(drop=True)
    user_folder = str(df.user.iloc[0]).zfill(3)
    transport_filepath = os.path.join(data_folder, user_folder, "labels.txt")
    if os.path.exists(transport_filepath):
        trans_df = pd.read_csv(transport_filepath, sep="\t")
        start_unix = pd.to_datetime(trans_df["Start Time"]).values.astype(int)
        end_unix = pd.to_datetime(trans_df["End Time"]).values.astype(int)
        trans_df = trans_df.rename(columns={"Transportation Mode": "transport"})
        old_length = len(df)

        user_unix = df["time"].values.astype(int)
        i, j = np.where((user_unix[:, None] >= start_unix) & (user_unix[:, None] <= end_unix))

        # sometimes we have time overlapping transportation labels.
        # those will be removed completly
        u, count = np.unique(i, return_counts=True)
        dup = u[count > 1]
        mask_duplicates = np.isin(i, dup)
        j = j[~mask_duplicates]
        i = i[~mask_duplicates]

        df = pd.concat([
            pd.concat([
                df.iloc[i, :].reset_index(drop=True),
                trans_df["transport"].iloc[j].reset_index(drop=True)
            ], axis=1),
            df.reset_index(drop=True).drop(i)],
            ignore_index=True, sort=False
        ).sort_values(by="time").reset_index(drop=True)

        assert len(df) == old_length, \
            "Dataframe length changed while merging transportation mode. This should not happen"

    return df



def import_and_preprocess_data(folderpath:str)->pd.DataFrame:
    data_folder = os.path.join(folderpath, "Geolife Trajectories 1.3", "Data")
    user_folders = next(os.walk(data_folder), (None, None, []))[1]
    gdfs = []

    for user_folder in tqdm(user_folders):

        traj_folder = os.path.join(data_folder, user_folder, "Trajectory")
        filenames = next(os.walk(traj_folder), (None, None, []))[2]
        filepaths = [os.path.join(traj_folder, filename) for filename in filenames if filename.endswith(".plt")]

        user_df = [
            pd.read_csv(filepath, sep=",", usecols=[0,1,3,4], header=5, names=["lat", "lon", "alt", "time"])
            for filepath in filepaths
        ]
        user_df = pd.concat(user_df)

        user_df["time"] = dt.datetime.fromisoformat("1899-12-30") + pd.to_timedelta(user_df["time"], unit="d")
        user_df = user_df.groupby(['time'],
            as_index=False).mean() # mean duplicate timestamps
        user_df["user"] = int(user_folder)
        user_df = user_df.sort_values(by="time")

        user_df = add_transportation_mode_col(user_df, data_folder)
        user_gdf = gpd.GeoDataFrame(user_df, geometry=gpd.points_from_xy(user_df.lon, user_df.lat), crs=utils.GPS_CRS)
        user_gdf = utils.remove_outliers(user_gdf.to_crs(GL_PROJ)).to_crs(utils.GPS_CRS)
        user_gdf["weak_sp"] = pd.Series(0.5, index=user_gdf.index) #0.5 is a placeholder value and is not used/masked in calculations
        user_gdf["weak_c"]  = pd.Series(0,   index=user_gdf.index)

        in_beijing_filter = is_inner_beijing(user_gdf)
        user_gdf_bj = user_gdf[in_beijing_filter].copy()
        user_gdf_no_bj = user_gdf[~in_beijing_filter].copy()

        if len(user_gdf_bj) > 0:
            # use OSM labels only for inner part of beijing, because otherwise data might be too large
            _, visited_amenities, _, user_gdf_bj = utils.enrich_with_osm_contex(user_gdf_bj)
            user_gdf_bj["weak_sp"], user_gdf_bj["weak_c"] = utils.weak_label(user_gdf_bj, AMENITIES_MEAN_AREA, visited_amenities)

            user_gdf = pd.concat([user_gdf_bj, user_gdf_no_bj]).sort_values(by="time").reset_index(drop=True)
        else:
            user_gdf = user_gdf_no_bj
        user_gdf = user_gdf.drop(columns=["lat", "lon"])
        user_gdf["alt"] = user_gdf["alt"] * 0.3048 # convert from feet to meters

        #append to complete dataframe
        gdfs.append(user_gdf)


    gdf = pd.concat(gdfs).reset_index(drop=True)

    #Convert float64 -> flaot32 columns to reduce memory allocation
    float64_cols = gdf.select_dtypes(include=['float64']).columns.values
    gdf[float64_cols] = gdf[float64_cols].astype("float32")

    # Identify separate trajectories and give each a unique ID
    gdf["track_id"] = utils.get_track_id(gdf)
    gdf.loc[gdf.track_id != gdf.track_id.shift(1), ["time_diff", "velocity", "distance"]] = 0 # by own definition
        
    return gdf


if __name__ == "__main__":
    folderpath = download()
    gdf = import_and_preprocess_data(folderpath)
    print("preprocessing finished")
    
    preproc_filepath = os.path.join(file_dir, 'tmp', 'preprocessed')
    os.makedirs(preproc_filepath, exist_ok=True)
    gl_preproc_filepath = os.path.join(file_dir, 'tmp', 'preprocessed', 'gl')
    gdf.to_pickle(gl_preproc_filepath + ".pkl")

    """
    kfold split for weak stay point labels (called `gl`) by their users.
    If it is not intended for a specific experiment to have a split based on users but rather to have a random split,
    then the easiest way to achieve this is to pass the complete dataset to src/experiment.py#start() with `train_data_path`
    and set the parameters `test_frac` and `test_k`, which makes the train/test split a random kfold split. For this work
    a random split is only necessary for the comparison with Dabriri et al., 2019 (Experiment 2)
    """
    splits = utils.kfold_split_by_users(gdf, lambda gdf: gdf.weak_c)
    kfold_folderpath = os.path.join(preproc_filepath, "gl_kfold")
    os.makedirs(kfold_folderpath, exist_ok=True)
    for i, (gdf_train, gdf_test) in enumerate(splits):
        gdf_train.to_pickle(os.path.join(kfold_folderpath, f"{i}_train.pkl"))
        gdf_test.to_pickle( os.path.join(kfold_folderpath, f"{i}_test.pkl"))
    
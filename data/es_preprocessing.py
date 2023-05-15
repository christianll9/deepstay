import os
import gzip
import shutil
import utils
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd

file_dir = os.path.dirname(os.path.abspath(__file__))

ES_CRS = "EPSG:32611"
AMENITIES_MEAN_AREA = 43092.96

FOLDERPATH_ABS_LOC = "ExtraSensory.per_uuid_absolute_location"
FOLDERPATH_FEA_LAB = "ExtraSensory.per_uuid_features_labels"
FOLDERPATH_ORIG_LAB = "ExtraSensory.per_uuid_original_labels"

def download_datasets(folderpath:str=None):
    if folderpath is None:
        folderpath = os.path.join(file_dir, 'tmp')
    URLS_ENDS = [
        "primary_data_files/ExtraSensory.per_uuid_features_labels.zip",
        "additional_data_files/ExtraSensory.per_uuid_original_labels.zip",
        "additional_data_files/ExtraSensory.per_uuid_absolute_location.zip"
    ]
    for url_end in URLS_ENDS:
        utils.download_dataset("http://extrasensory.ucsd.edu/data/" + url_end, folderpath)
    return folderpath

def shorten_prefix(label:str) -> str:
    """Returns the label with shorter or none prefix like 'label:'"""
    if label.startswith("label:"):
        return label[6:]
    elif label.startswith("original_label:"):
        return "orig_" + label[15:]
    return label

def is_in_la_or_sd(df:pd.DataFrame)->pd.DataFrame:
    """
    Filter data on Los Angeles or San Diego where 97% of the data lays
    """
    return (df.longitude > -117.3) &\
           (df.longitude < -116.9) &\
           (df.latitude > 32.6) &\
           (df.latitude < 32.9)

def extract_gz_files(folderpath:str):

    for rel_path in [FOLDERPATH_ABS_LOC, FOLDERPATH_FEA_LAB, FOLDERPATH_ORIG_LAB]:
        count = 0
        abs_folderpath = os.path.join(folderpath, rel_path)
        filenames = next(os.walk(abs_folderpath), (None, None, []))[2]  # [] if no file
        # if files are gz files and cooresponding csv file is not yet extracted
        gz_file2extr = [filename for filename in filenames if filename[-7:] == ".csv.gz" and not filename[:-3] in filenames]
        for gz_file in gz_file2extr:
            filepath = os.path.join(abs_folderpath, gz_file)
            with gzip.open(filepath, 'rb') as f_in:
                with open(filepath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    count += 1

        if count > 0:
            print(f"Extracted {count} files from {folderpath}")
        else:
            print(f"All files in {folderpath} are already extracted")


def stay_labels(df: gpd.GeoDataFrame) -> pd.Series:
    """
    Mapping from ES activity labels to stay labels
    Note: ELEVATOR seems to be problematic and include way to much movement, so it will not be used
    """

    # these "clear" stay points can still be outdone by a transportation label
    plausible_stay_points = [
        "label:LOC_home", "label:LYING_DOWN", "label:SLEEPING", "label:COMPUTER_WORK", "label:LOC_main_workplace",
        "label:WATCHING_TV", "label:IN_CLASS", "label:IN_A_MEETING", "label:COOKING", "label:LAB_WORK", "label:CLEANING",  "label:GROOMING", "label:TOILET",
        "label:DRESSING", "label:FIX_restaurant", "label:BATHING_-_SHOWER", "label:AT_A_PARTY", "label:WASHING_DISHES", "label:AT_THE_GYM",
        "label:STAIRS_-_GOING_UP", "label:STAIRS_-_GOING_DOWN", "label:DOING_LAUNDRY", "label:AT_A_BAR",
        "original_label:JUMPING", "original_label:BATHING_-_BATH", "original_label:AT_A_SPORTS_EVENT", "original_label:YOGA", "original_label:LIFTING_WEIGHTS",
        "original_label:TRANSFER_-_STAND_TO_BED", "original_label:AT_THE_POOL", "original_label:PLAYING_MUSICAL_INSTRUMENT", "original_label:AT_A_CONCERT",
        "original_label:VACUUMING", "original_label:TRANSFER_-_BED_TO_STAND", "original_label:GARDENING", "original_label:RAKING_LEAVES", "original_label:TREADMILL",
        "original_label:ELLIPTICAL_MACHINE", "original_label:STATIONARY_BIKE", "original_label:DANCING", "original_label:PLAYING_BASEBALL"
    ]
    plausible_stay_points = [label for label in plausible_stay_points if label in df]

    otherwise_stay_points = ["label:PHONE_ON_TABLE", "original_label:WRITTEN_WORK", "label:OR_indoors", "label:AT_SCHOOL"]

    clearly_no_stay_points = [
        "label:DRIVE_-_I_M_THE_DRIVER", "label:BICYCLING", "label:DRIVE_-_I_M_A_PASSENGER", "label:ON_A_BUS", "label:FIX_running",
        "original_label:ON_A_TRAIN", "original_label:MOTORBIKE"
    ]
    plausibly_no_stay_points = ["label:FIX_walking"]

    sps, otherwise_sps, clearly_no_sps, plausibly_no_sps = \
    [np.any(df[corresp_labels], axis=1) for corresp_labels in [plausible_stay_points, otherwise_stay_points, clearly_no_stay_points, plausibly_no_stay_points]]

    # rules can be improved
    no_sps = clearly_no_sps | plausibly_no_sps
    stay_points = pd.Series(
        np.select([(sps & (~no_sps)), (no_sps & (~sps)), (otherwise_sps & (~no_sps))], [1, 0, 1], default=np.nan),
        index=df.index
    )


    # remove likely missclassified stay points:
    # speed of stay points should never exceed speed average of non stays

    # speed average of non stays
    speed_thresh = df[stay_points == 0]["mean_speed"].mean()

    # speed average of no stay points based on GPS diff (since mean_speed is often NaN)
    gps_diff_speed_thresh = df[stay_points == 0]["velocity"].mean()

    stay_points = stay_points.where(
        (stay_points == 0) | (df["mean_speed"] <= speed_thresh) | (df["mean_speed"].isna() & (df["velocity"] <= gps_diff_speed_thresh))
    ) # if stay point & speed > speed average => nan
    return stay_points



def import_and_preprocess_data(folderpath:str)->pd.DataFrame:
    extract_gz_files(folderpath)

    filenames = next(os.walk(os.path.join(folderpath, FOLDERPATH_ABS_LOC)), (None, None, []))[2]  # [] if no file
    uuids = [filename[:36] for filename in filenames if filename[-4:] == ".csv"]
    gdfs = []

    for uuid in tqdm(uuids):
        #read files
        df_loc = pd.read_csv(os.path.join(
            folderpath, FOLDERPATH_ABS_LOC, f"{uuid}.absolute_locations.csv"))
        df_lab = pd.read_csv(os.path.join(
            folderpath, FOLDERPATH_FEA_LAB, f"{uuid}.features_labels.csv"))
        df_orig = pd.read_csv(os.path.join(
            folderpath, FOLDERPATH_ORIG_LAB, f"{uuid}.original_labels.csv"))

        # filter columns
        filtered_substr = ["raw", "proc_gyro", "watch_acceleration", "watch_heading", "audio", "discrete", "lf_measurements"]
        df_lab = df_lab.iloc[:,[all([substr not in col_name for substr in filtered_substr]) for col_name in df_lab.columns]]
        df_orig = df_orig.drop(columns=[label for label in df_orig.columns if label[len("original_"):] in df_lab.columns] + ["label_source"])
        df_orig = df_orig.where(df_orig > 0)

        # merge
        user_df = pd.merge(df_lab,df_loc,on="timestamp")
        user_df = pd.merge(user_df, df_orig, on="timestamp")
        assert len(user_df) == len(df_lab) and len(df_lab) == len(df_loc) and len(df_loc) == len(df_orig), "merge did not work as expected"

        #format and aggregate
        user_df["timestamp"] = pd.to_datetime(user_df["timestamp"], unit='s', origin='unix')
        user_df = user_df.rename(columns={"timestamp": "time"})
        assert (user_df["time"] - user_df["time"].shift() > pd.to_timedelta(0))[1:].all(), "timestamps are not strictly increasing"
        labels = [label for label in user_df.columns if "label:" in label]
        summarize_labels = lambda vals: [shorten_prefix(label) for label, val in zip(labels, vals) if val == 1]
        user_df["label_summary"] = user_df[labels].apply(summarize_labels, axis=1)
        user_df["user"] = uuid

        user_df = user_df[~user_df.longitude.isna()] # remove nan points
        user_gdf = gpd.GeoDataFrame(user_df,
            geometry=gpd.points_from_xy(user_df.longitude, user_df.latitude), crs=utils.GPS_CRS)
        user_gdf = utils.remove_outliers(user_gdf.to_crs(ES_CRS)).to_crs(utils.GPS_CRS)
        user_gdf["mean_speed"] = (user_gdf["location:min_speed"] + user_gdf["location:max_speed"])/2
        user_gdf["weak_sp"] = pd.Series(0.5, index=user_gdf.index) #0.5 is a placeholder value and is not used/masked in calculations
        user_gdf["weak_c"]  = pd.Series(0,   index=user_gdf.index)

        in_la_or_sd_filter = is_in_la_or_sd(user_gdf)
        user_gdf_in = user_gdf[in_la_or_sd_filter].copy()
        user_gdf_out = user_gdf[~in_la_or_sd_filter].copy()

        if len(user_gdf_in) > 0:
            # use OSM weak labels only for LA and San Diego, because otherwise extracted data will be too large to process
            _, visited_amenities, _, user_gdf_in = utils.enrich_with_osm_contex(user_gdf_in)
            user_gdf_in["weak_sp"], user_gdf_in["weak_c"] = utils.weak_label(user_gdf_in, AMENITIES_MEAN_AREA, visited_amenities)

            user_gdf = pd.concat([user_gdf_in, user_gdf_out]).sort_values(by="time").reset_index(drop=True)
        else:
            user_gdf = user_gdf_out
        gdfs.append(user_gdf)

    #combine all users into one gdf
    gdf = pd.concat(gdfs, ignore_index=True)

    # remove labels, with not a single positive sample
    not_used_labels = gdf[labels].columns[(gdf[labels].sum() < 1)]
    gdf = gdf.drop(columns=not_used_labels)
    labels = [label for label in labels if label not in not_used_labels]

    #Convert float64 -> flaot32 columns to reduce memory allocation
    float64_cols = gdf.select_dtypes(include=['float64']).columns.values
    gdf[float64_cols] = gdf[float64_cols].astype("float32")

    # remove point with missing locations
    gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]

    # Attach mapped labels
    gdf["stay_point"] = stay_labels(gdf)

    # Identify separate trajectories and give each a unique ID
    gdf["track_id"] = utils.get_track_id(gdf)
    gdf.loc[gdf.track_id != gdf.track_id.shift(1), ["time_diff", "velocity", "distance"]] = 0 # by own definition

    return gdf


if __name__ == "__main__":
    folderpath = download_datasets()
    gdf = import_and_preprocess_data(folderpath)
    print("preprocessing finished")
    
    preproc_filepath = os.path.join(file_dir, 'tmp', 'preprocessed')
    os.makedirs(preproc_filepath, exist_ok=True)
    preproc_filepath = os.path.join(preproc_filepath, "es")
    gdf.to_pickle(preproc_filepath + ".pkl")

    # kfold split
    splits = utils.kfold_split_by_users(gdf, lambda gdf: ~gdf.stay_point.isna())
    kfold_folderpath = preproc_filepath + "_kfold"
    os.makedirs(kfold_folderpath, exist_ok=True)
    for i, (gdf_train, gdf_test) in enumerate(splits):
        gdf_train.to_pickle(os.path.join(kfold_folderpath, f"{i}_train.pkl"))
        gdf_test.to_pickle( os.path.join(kfold_folderpath, f"{i}_test.pkl"))
    
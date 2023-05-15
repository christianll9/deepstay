from typing import Callable
from torch.utils.data import Dataset, Subset
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch import default_generator, Tensor
from abc import abstractmethod


def rot_traj(x, angle=None):
    """
    Randomly rotate trajectory `x` by axis for data augmentation
    Formular source: https://en.wikipedia.org/wiki/Rotation_of_axes#math_5
    x: Trajectory of shape: ((...,) sequence length, 2) | tested on 3dim and 2dim
    """
    if angle is None:
        angle = (torch.rand(x.shape[:-2], device=x.device)*2*torch.pi)[..., None]
    sine = torch.sin(angle)
    cosine = torch.cos(angle)
    return torch.cat([
        ( x[...,0] * cosine + x[...,1] * sine  )[..., None],
        (-x[...,0] * sine   + x[...,1] * cosine)[..., None]
    ], -1), angle


def interpolate(t:np.ndarray, x:np.ndarray, y:np.ndarray, interp_sec:float,
    time_diff_col_idx:int=None, y_interp_meth:str="linear"):
    """
    t: orginal time in nanoseconds
    time_diff_col_idx: column index of time_diff feature within x, None if time_diff not in x
    """
    #interval = interp_sec*1E9
    t = ((t-t[0])/1E9).astype("float32") #nanoseconds -> seconds
    t_new = np.arange(t.min(), t.max(), interp_sec, dtype="float32")

    if y_interp_meth == "linear":
        y = interp2darr(t_new, t, y).astype("float32")
    elif y_interp_meth == "nearest":
        y = nearest_interp(t_new, t, y.astype("float32"))
    else:
        raise Exception(f"Unknown interploation method {y_interp_meth}")

    # weight individual points heigher, if there are more near to original points
    p_weights = ((t[:,None] < t_new[None,:]+interp_sec/2) & (t[:,None] >= t_new[None,:]-interp_sec/2)).sum(0).astype("uint16")
    p_weights[-1] += (t >= t_new[-1]-interp_sec/2).sum()

    if time_diff_col_idx is None:
        x = interp2darr(t_new, t, x)
    else:
        more_feat_available = time_diff_col_idx < len(x.T)-1
        x = np.hstack([
            interp2darr(t_new, t, x[:,:time_diff_col_idx]),
            # new time_diff feature
            np.array(([0] if len(t_new) != 0 else []) + (len(t_new)-1)*[interp_sec], dtype=np.float32)[:,None]
        ]).astype("float32")
        if more_feat_available:
            x = np.hstack([
                x,
                interp2darr(t_new, t, x[:,time_diff_col_idx+1:])
            ]).astype("float32")

    return x, y, p_weights


def interp2darr(t_new:np.ndarray, t_src:np.ndarray, arr2d:np.ndarray) -> np.ndarray:
    """
    intperpolate all feature columns with np.interp(t_new, t_src, column)
    
    probably not the most efficient way, but it works (todo: replace for-loop
    with apply function)
    """
    return np.hstack(
        [np.interp(t_new, t_src, arr2d[:,col])[:, None] for col in range(arr2d.shape[-1])]
    )

def nearest_interp(t_new:np.ndarray, t_src:np.ndarray, arr2d:np.ndarray):
    """Src: https://stackoverflow.com/a/21003629/11227118 """
    idx = np.abs(t_src - t_new[:,None])
    return arr2d[idx.argmin(axis=1)]


def split_gdf_track2np_seqs(track:gpd.GeoDataFrame, id:int, seq_length:int, y:np.ndarray, y_weights:np.ndarray=None,
    interp_sec:int=None, add_features:list=[], include_remain:bool=False, y_interp_meth:str="linear"):
    """
    GeoDataFrame into numpy sequence arrays
    include_remain: if true and remainings exist, last sequence contains data[-seq_length:]
                    (points_weights are 0 for already seen data in last sequence)
    """

    x = np.hstack((
        track.geometry.x.to_numpy('float32')[:, None],
        track.geometry.y.to_numpy('float32')[:, None],
        track["velocity"].to_numpy()[:, None], #necessary for 2. ssl task
        *[
            track[add_feature].to_numpy()[:, None]
            for add_feature in add_features if add_feature != "velocity"
        ]
    ))

    if len(x) < 2:
        interp_sec = None # nothing to interpolate

    # Linear interpolation (Moreau et al., 2021)
    if interp_sec is None:
        y_weights = y_weights if y_weights is not None else np.ones(len(x))
        x_weights = np.ones(len(x))
    else:
        assert y_weights is None, "interpolated label weights are not supported yet"
        t = track.time.astype(np.int64).to_numpy()
        if "time_diff" in add_features:
            time_diff_col_idx = 3 + add_features.index("time_diff")
        else:
            time_diff_col_idx = None
        x, y, x_weights = interpolate(t, x, y, interp_sec, time_diff_col_idx=time_diff_col_idx, y_interp_meth=y_interp_meth)
        y_weights = np.copy(x_weights)

    num_seqs = len(x)//seq_length

    if len(x)%seq_length == 0: # if whole track fits in sequences with seq_length
        num_seqs -= 1        # leave one sequence for self-supervision task
    
    if num_seqs <= 0:
        # not enough data in this track
        if include_remain and len(x)>0:
            # Return part of sequence even thought it does not fill a hole sequence.
            # Fill rest of sequence with nan
            x_nan = np.full((1, seq_length, x.shape[-1]), np.nan)
            y_nan = np.full((1, seq_length, y.shape[-1]), np.nan)
            x_weights_0 = np.zeros((1, seq_length))
            y_weights_0 = np.zeros((1, seq_length))
            x_next = np.full((1, 3), np.nan)
            x_nan[0, :len(x)] = x
            y_nan[0, :len(y)] = y
            x_weights_0[0, :len(x_weights)] = x_weights
            y_weights_0[0, :len(y_weights)] = y_weights
            id = np.full(len(x_nan),id)
            return x_nan, y_nan, x_weights_0, y_weights_0, x_next, id
        else:
            #return empty arrays with proper shape for concatenating later
            return np.empty((0, seq_length, 2+len(add_features))), np.empty((0, seq_length, y.shape[-1])),\
                np.empty((0, seq_length)), np.empty((0, seq_length)), np.empty((0, 3)), np.empty((0))

    x_next = x[seq_length::seq_length, :3]

    if include_remain:
        x_last = x[-seq_length:]
        y_last = y[-seq_length:]
        x_weights_last = np.copy(x_weights[-seq_length:])
        y_weights_last = np.copy(y_weights[-seq_length:])
        # this data is already included in data below, so don't weight it twice
        x_weights_last[:(num_seqs+1)*seq_length-len(x)] = 0
        y_weights_last[:(num_seqs+1)*seq_length-len(x)] = 0

    x = x[:num_seqs*seq_length].reshape(-1, seq_length, x.shape[-1])
    y = y[:num_seqs*seq_length].reshape(-1, seq_length, y.shape[-1])
    x_weights = x_weights[:num_seqs*seq_length].reshape(-1, seq_length)
    y_weights = y_weights[:num_seqs*seq_length].reshape(-1, seq_length)

    if include_remain:
        x = np.append(x, x_last[None, ...], axis=0)
        y = np.append(y, y_last[None, ...], axis=0)
        x_weights = np.append(x_weights, x_weights_last[None, ...], axis=0)
        y_weights = np.append(y_weights, y_weights_last[None, ...], axis=0)
        x_next = np.append(x_next, np.full([1, x_next.shape[-1]], np.nan), axis=0)

    # remove purely interpolated sequences
    real_points_included = x_weights.sum(-1) > 0
    x, y, x_weights, y_weights, x_next = [elem[real_points_included] for elem in [x, y, x_weights, y_weights, x_next]]

    if not "velocity" in add_features:
        x = np.delete(x, 2, -1) #exclude it again

    id = np.full(len(x), id)

    return x, y, x_weights, y_weights, x_next, id


######################## Datasets #################################

class Traj_Dataset(Dataset):
    """
    Values are oriented by Dabiri et al.
    interp_sec:          Linear interpolation, Moreau et al. (2021) used 2 sec
    label_func:          Input: GeoDataFrame track, Output: Labels as numpy array
    """
    def __init__(self, preprocessed_pkl_filepath:str, epsg:int, label_func:Callable, 
        seq_length:int=256, interp_sec:float=None, add_features:list=["time_diff", "velocity"],
        data_filter:Callable=None, norm_seq_pos:bool=True, weight_func:Callable=lambda _: None,
        split_kwargs:dict={}):

        with open(preprocessed_pkl_filepath, "rb") as f:
            data = pickle.load(f)

        if data_filter is not None:
            data = data_filter(data)
        data = data.to_crs(epsg=epsg) # Project to UTM zone 50N

        x, y, w_x, w_y, x_next, id = zip(*[
            split_gdf_track2np_seqs(track, track_id, seq_length, y=label_func(track), y_weights=weight_func(track),
                interp_sec=interp_sec, add_features=add_features, **split_kwargs)
            for track_id, track in data.groupby("track_id")
        ])
        del data
        x, y, w_x, w_y, x_next, id = [
            torch.tensor(np.concatenate(elem).astype(np.float32))
            for elem in [x, y, w_x, w_y, x_next, id]
        ]

        self.x = x
        self.y = y
        self.w_x = w_x # sample weights
        self.w_y = w_y # label weights
        self.x_next = x_next
        self.id = id
        if norm_seq_pos:
            self.norm_sequence_pos()

    @staticmethod
    @abstractmethod
    def comp_fun(model_output:Tensor, target:Tensor) -> Tensor:
        """
        Compares the model output with the target tensor and returns boolean Tensor.
        """

    def norm_sequence_pos(self):
        """
        Normalize sequence position by removing the offset to prevent memorization of regions
        """
        seq_mean = self.x[:, :, :2].nanmean(1)
        self.x[:, :, :2] = self.x[:, :, :2] - seq_mean[:, None]
        self.x_next[:, :2] = self.x_next[:, :2] - seq_mean

    
    def split(self, val_frac:float=0.2, generator=default_generator, val_k:int=0,
        test_frac:float=None, test_k:int=0) -> tuple[Subset]:
        """
        Splits the dataset in trainval and test. Then again in train and val. This code needs to be
        rewritten in a more readable, elegant, and recursive way. `k` can index the split segment
        for k-fold cross-validation (thus generator needs to be initialized with same seed)

        `test_frac` is the fraction of the whole dataset, val_frac is only fraction of the remaining
        training set
        """
        no_test = test_frac is None
        if no_test:
            test_frac = 0
        test_length = round(test_frac*len(self))
        trainval_length = len(self) - test_length
        test_start = test_length*test_k
        val_length = round(val_frac*trainval_length)
        val_start = val_length*val_k
        if val_start >= test_start:
            val_start = val_start + test_length
        test_end = test_start + test_length


        train1_length = min(test_start, val_start)
        val1_length = min(val_length, test_start-train1_length)
        val1_end = train1_length + val1_length
        train2_length = test_start - val1_end
        val2_length = val_length - val1_length
        train3_length = max(0, val_start-test_end)
        train4_length = len(self) - train3_length - train2_length - train1_length - val2_length - val1_length - test_length

        if train4_length < 0:
            if val2_length > 0:
                val2_length += train4_length
                assert val2_length >= 0, "Negative split length"
            elif test_length > 0:
                test_length += train4_length
                assert test_length >= 0, "Negative split length"
            else:
                val1_length += train4_length
                assert val1_length >= 0, "Negative split length"
            train4_length = 0

        train1, val1, train2, test, train3, val2, train4 = torch.utils.data.random_split(self,
            [train1_length, val1_length, train2_length, test_length, train3_length, val2_length, train4_length], generator=generator)
        train = Subset(self, train1.indices + train2.indices + train3.indices + train4.indices)
        val = Subset(self, val1.indices + val2.indices)
        if no_test:
            return train, val
        else:
            return train, val, test


    def get_norm_params_of_train_data(self, train_idxs):
        """
        returns normalization parameters based on training data
        normalizing data after train/test split to avoid data leakage (mean, std information) from test split
        feature1-2: coords
        feature>2: additional features like velocity, time_diff, ...
        """
        coords_std = self.x[train_idxs, :, :2].std()

        #features like velocity, time_diff
        feat_mean = self.x[train_idxs,:,2:].mean((0,1))
        feat_std  = self.x[train_idxs,:,2:].std( (0,1))

        return coords_std, feat_mean, feat_std

    def normalize(self, coords_std, feat_mean, feat_std):
        self.x[:, :, :2] = self.x[:, :, :2]/coords_std
        self.x_next = self.x_next/coords_std
        #features like velocity, time_diff, ...
        self.x[:,:,2:] = self.x[:,:,2:] - feat_mean[None, None, :]
        self.x[:,:,2:] = self.x[:,:,2:] / feat_std[None, None, :]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return tuple(elem[idx] for elem in [self.x, self.y, self.w_x, self.w_y, self.x_next])


######################## Transportation Mode Datasets #################################

class TM_Dataset(Traj_Dataset):
    """
    Here, labels are different transportation modes (see `self.classes`)
    """
    def __init__(self, *args, **kwargs):

        # categories oriented on Dabiri et al.:
        self.classes = ['bike', 'bus', 'walk', 'drive', 'train_']
        def label_func(track):
            track["bike"] = track.transport == "bike"
            track["bus"] = track.transport == "bus"
            track["walk"] = track.transport == "walk"
            track["drive"] = track.transport.isin(["car", "taxi"])
            track["train_"] = track.transport.isin(["train", "subway"])
            return track[self.classes].to_numpy()

        super(TM_Dataset, self).__init__(
            *args, **kwargs, epsg=32650, label_func=label_func,
            weight_func=lambda track: track[self.classes].sum(1).to_numpy())

    @staticmethod
    def comp_fun(model_output:Tensor, target:Tensor) -> Tensor:
        return model_output.softmax(dim=-1).argmax(dim=-1) == target.argmax(dim=-1)


######################## Stay Point Datasets #################################

class StayPoint_Dataset(Traj_Dataset):
    def split(self, val_frac:float=0.2, *args, **kwargs):
        """checks roughly for a stratified split"""
        train_set, val_set = super().split(val_frac, *args, **kwargs)
        if 0 < val_frac and val_frac < 1:
            class_ratio_train =  (self.y[train_set.indices] < 0.5).sum()/(self.y[train_set.indices] > 0.5).sum()
            class_ratio_val =  (self.y[val_set.indices] < 0.5).sum()/(self.y[val_set.indices] > 0.5).sum()
            assert (0.5 < class_ratio_train/class_ratio_val) and (class_ratio_train/class_ratio_val < 1.5), \
                f"train and validation split differ significant in non-stay_point/stay_point ratio." \
                    + f" (train: {class_ratio_train}, val: {class_ratio_val})"
        return train_set, val_set


    @staticmethod
    def comp_fun(model_output:Tensor, target:Tensor) -> Tensor:
        return ((model_output > 0.5) == (target > 0.5))[..., 0]


class GL_Dataset(StayPoint_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, epsg=32650,
            label_func =lambda track: track["weak_sp"].to_numpy()[:,None],
            weight_func=lambda track: track["weak_c"].to_numpy())

class ES_Dataset(StayPoint_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, epsg=32611,
            label_func=lambda track: track["stay_point"].to_numpy()[:,None])
        self.w_y[torch.isnan(self.y)[..., 0]] = 0
        self.y[torch.isnan(self.y)] = -1E6 #for samples where w_y=0 to ensure w_y * arbitrary_fun(y) = 0 instead of nan

import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.geometry as geom
from scipy import special
import math

###################### Helper functions ######################

def _merge_clusters(*args):
    """
    Helper function for merging clusters
    """
    return sorted(list(set(sum(args,[]))))

def _add_clusters2traj(C:list, traj:pd.DataFrame, cluster_ids:list=None) -> pd.Series:
    """
    Helper function to add cluster labels to trajectory. Label points outside a cluster unique and negative.
    """
    # negative idx indicates non-stay (each non-stay point has a unique negative cluser label)
    clusters = -(traj.index).to_series(name="cluster")-1
    if cluster_ids is None:
        cluster_ids = list(range(1, len(C)+1))
    for c_id, c in zip(cluster_ids, C):
        assert (clusters.loc[c] < 0).all(),\
            f"Error: At least one point in cluster c{c_id} is already assigned to another cluster."
        clusters.loc[c] = c_id

    return clusters

def _duration(point_idxs:list, traj:gpd.GeoDataFrame) -> float:
    """
    Helper function that returns duration in seconds
    """
    if len(point_idxs) > 1:
        return (traj.loc[point_idxs].time.max() - traj.loc[point_idxs].time.min()).total_seconds()
    else:
        return 0

def _centroid(point_idxs:list, traj:gpd.GeoDataFrame) -> geom.Point:
    """
    Helper function to calculate centroid of a set of points from a trajectory
    """
    return geom.Point(traj.loc[point_idxs].geometry.x.mean(), traj.loc[point_idxs].geometry.y.mean())



###################### Stay Region Extraction ######################



def d_star(traj: gpd.GeoDataFrame, d_max:float=30, q:int=30, t_min:float=135, T_stay:float=300) -> pd.Series:
    """
    Nishida et al. (2015)
    traj: Location trajectory as GeoDataFrame with ascending timestamps
    Naive and basic implementation. Lot of run performance optimization possible
    cluster is negative for non-stay points
    """

    C_global = []
    for _, track in traj.groupby("track_id"): #this line is not within the original publication
    
        C = [] # clusters
        wind = []
        wind_neigh = [] # list of neighborhood lists of window points

        for i in track.index:
            #Create empty neighborhood
            wind_neigh.append([])
            for idx, j in enumerate(wind):
                if track.loc[i].geometry.distance(track.loc[j].geometry) <= d_max:
                    # Add p_i to N(p_j)
                    wind_neigh[idx].append(i)
                    # Add p_j to N(p_i)
                    wind_neigh[-1].append(j)

            wind.append(i)
            wind_neigh[-1].append(i)

            if i >= q:
                N_p_iq = wind_neigh[0] # N(p_{i-q})

                # Shift p_{i-q} from sliding window
                wind = wind[-q:]
                wind_neigh = wind_neigh[-q:]

                if _duration(N_p_iq, track) >= t_min:
                    dur_joinables = [c_idx for c_idx, c in enumerate(C) if any([j in N_p_iq for j in c])]
                    if len(dur_joinables) > 0:
                        # Merge N(p_{i−q}) and all its duration-joinable clusters.
                        C[dur_joinables[0]] = _merge_clusters(N_p_iq, *[C[idx] for idx in dur_joinables])
                        for idx in reversed(sorted(dur_joinables[1:])):
                            del C[idx] # remove all other merged clusters
                    else:
                        # Create a new cluster based on N(p_{i−q}).
                        C.append(N_p_iq)
        C = [c for c in C if _duration(c, track) >= T_stay]
        C_global += C

    clusters = _add_clusters2traj(C_global, traj)
    return clusters


def cb_smot(traj:gpd.GeoDataFrame, t_min:float=300, a:float=0.5) -> pd.Series:
    """
    Stay region Extraction algorithm from Palma et al., 2008
    In this implementation, we do not use any prior knowledge of an application A.
    Input:
        a:      area for the quantile function
                (approximate proportion of points that generate potential stops
                in relation to the total amount of points, value lies between 0 and 1)
        t_min:  minimun time for clustering
    Output:
        S: set of stops
        M: set of moves
    """

    def linear_neighborhood(traj:gpd.GeoDataFrame, distances:gpd.GeoSeries, k:int, eps:float) -> list:
        distances.iloc[0] = 0 # for simpler calculation

        # lin dist to next points (k excluded)
        dist2next = (distances.loc[k:].cumsum() - distances.loc[k]).loc[k+1:]
        # lin dist to previous points (k included)
        dist2prev = (distances.shift(-1, fill_value=0).loc[k::-1].cumsum() - distances.shift(-1, fill_value=0).loc[k])[::-1]

        lin_dist_to_k = pd.concat([dist2prev, dist2next])
        return traj.index[lin_dist_to_k <= eps].to_list()

    def quantile(μ, std):
        eps = μ + std * math.sqrt(2) * special.erfinv(2*a-1)
        assert eps >= μ/2, f"eps lies below μ/2. eps={eps}, μ/2={μ/2} | This is probably not intended. Use a higher value for a"
        return eps

    # compute the clusters
    C = [] #set clusters as empty

    for _, track in traj.groupby("track_id"):
        if len(track) < 3: #need at least 3 points to calculate std for distances of 1 point to all others 
            continue
        distances = track.distance(track.shift(1))
        μ = distances.mean()
        std = distances.std()
        processed = pd.Series(False, index=track.index)

        eps = quantile(μ, std)
        for p in track.index:
            if not processed.loc[p]:
                neighbors = [n for n in linear_neighborhood(track, distances, p, eps) if not processed.loc[n]]
                if _duration(neighbors, track) >= t_min:
                    additional_neighbors = []
                    for n in neighbors:
                        #add to neighbors every unprocessed point in linear_neighborhood
                        additional_neighbors += [
                            n_n for n_n in linear_neighborhood(track, distances, n, eps)
                            if not processed.loc[n_n] and n_n not in neighbors + additional_neighbors
                        ]
                    #set all points in neighbors as processed
                    neighbors += additional_neighbors
                    processed.loc[neighbors] = True
                    C.append(neighbors)

    # find stops and moves
    clusters = _add_clusters2traj(C, traj)
    return clusters


def kang2004_sr(traj:gpd.GeoDataFrame, d_max:float=40, t_min:float=300) -> pd.Series:
    """
    Stay region extraction based on Kang et al., 2004
    input: measured locations traj
    state:  current cluster c,
            pending location ploc,
            significant places C
    """
    C = []

    for _, track in traj.groupby("track_id"): #this line is not within the original publication

        c = []
        ploc = None
        c_mean = None

        for i in track.index:
            # Time-based clustering algorithm (Kang et al., 2004 - Table 1)
            if c_mean is not None and track.loc[i].geometry.distance(c_mean) < d_max:
                # add loc to c
                c.append(i)
                c_mean = _centroid(c, track)  # can be optimized with moving average
                ploc = None
            else:
                if ploc is not None:
                    if _duration(c, track) > t_min:
                        # check if c can be merged with any of the already existing C
                        C.append(c)
                    c = [ploc]
                    c_mean = track.loc[ploc].geometry
                    if track.loc[i].geometry.distance(c_mean) < d_max:
                        # add loc to cl
                        c.append(i)
                        c_mean =  _centroid(c, track)
                        ploc = None
                    else:
                        ploc = i
                else:
                    ploc = i

    clusters = _add_clusters2traj(C, traj)
    return clusters


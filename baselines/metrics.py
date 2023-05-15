from typing import Callable
import pandas as pd
import geopandas as gpd
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

def stay_accuracy(ground_truth: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate the accuracy of a stay region prediction.
    predicted: Series of predicted stay regions
    ground_truth: Series of stay regions
    """
    return pd.concat([predicted, ground_truth], axis=1)[ground_truth.notna()] \
        .astype(bool).nunique(axis=1).eq(1).mean()


def stay_conf_matrix(ground_truth: pd.Series, predicted: pd.Series) -> list[list[float]]:
    """
    Calculate the confusion matrix of a stay point prediction.
    """
    df = pd.concat([ground_truth, predicted], axis=1)[ground_truth.notna()].astype(bool)
    return confusion_matrix(df.iloc[:,0], df.iloc[:,1]).tolist()


def nonstay_f1(ground_truth: pd.Series, predicted: pd.Series, pos_label=False) -> float:
    """
    Calculate the F1 score of a non-stay point (`False` in series data) prediction.
    predicted: Boolean series of stay points 
    ground_truth: Series of stay points
    """
    df = pd.concat([ground_truth, predicted], axis=1)[ground_truth.notna()].astype(bool)
    return f1_score(df.iloc[:,0], df.iloc[:,1], pos_label=pos_label)

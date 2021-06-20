import os
import pandas as pd
import numpy as np


CAT_FEATURE_PREPROCESSING = ["OneHotEncoding", "OrdinalEncoder"]
NUM_FEATURE_PREPROCESSING = ["Normalize"]


def read_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def generate_random_dataset(cat_features=5, num_features=10, bin_features=2, n_sample=100, cat_options=None) -> pd.DataFrame:
    MIN_NUM_CATS = 2
    MAX_NUM_CATS = 8
    frame_list = []
    for idx in range(num_features):
        mean = np.random.random()
        deviation = np.random.random() + 1
        name = f"Num_{idx + 1}"
        num_frame = generate_random_numerical(mean, deviation, n_sample, name)
        frame_list.append(num_frame)
    for idx in range(cat_features):
        if cat_options is None:
            num_cats = np.random.randint(MIN_NUM_CATS, MAX_NUM_CATS)
        else:
            num_cats = cat_options[idx]
        bias = np.random.randint(0, 4)
        name = f"Cat_{idx + 1}"
        cat_frame = generate_random_categorical(num_cats, n_sample, shifting=bias, name=name)
        frame_list.append(cat_frame)
    for idx in range(bin_features):
        num_cats = 2
        name = f"Bin_{idx + 1}"
        bin_frame = generate_random_categorical(num_cats, n_sample, name=name)
        frame_list.append(bin_frame)
    num_cats = 4
    name = "Target"
    trg_frame = generate_random_categorical(num_cats, n_sample, name=name)
    frame_list.append(trg_frame)
    return pd.concat(frame_list, axis=1)


def generate_random_categorical(n_values=5, length=100, p=None, shifting=0, name=None) -> pd.Series:
    probe = [idx for idx in range(n_values)]
    list_val = [shifting + np.random.choice(probe, 1, p=p)[0] for _ in range(length)]
    return pd.Series(list_val, name=name)


def generate_random_numerical(mean=0, deviation=1, length=100, name=None) -> pd.Series:
    list_val = [np.random.normal(mean, deviation) for _ in range(length)]
    return pd.Series(list_val, name=name)


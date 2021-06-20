from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

import data as data
import classes as cls
import features as ftr


def train(x_train, y_train, params: cls.AttemptParams):
    model = None
    if params.model == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif params.model == "LogisticRegression":
        model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


def save_model(model, params: cls.AttemptParams):
    model_path = os.path.join(params.model_output_path, params.model + ".pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


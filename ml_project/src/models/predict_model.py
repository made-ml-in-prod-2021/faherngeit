import pickle
import os

import src.classes as cls


def load_model(params: cls.AttemptParams):
    model_path = os.path.join(params.model_output_path, params.model + '.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def evaluate(model, x_test):
    return model.predict(x_test)

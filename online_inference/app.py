import pickle
import logging
import os
from typing import List, Union
import time

from fastapi import FastAPI, BackgroundTasks, Response, status
from fastapi.logger import logger
import uvicorn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from pydantic import BaseModel, conlist

import classes

PATH_TO_MODEL = os.getenv("PATH_TO_MODEL") #"RandomForestClassifier.pkl"
PATH_TO_DESCRIPTION = os.getenv("PATH_TO_DESC")# "another_train.yaml"
MAX_RESPONSE = 10

uvicorn_logger = logging.getLogger("uvicorn")
logger.handlers = uvicorn_logger.handlers
logger.setLevel(logging.INFO)

app = FastAPI()
model = None
desc = None
transformer = None
crash = False
counter = 0

class DataEntity(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=10, max_items=80)]
    features: List[str]


class PredictResponce(BaseModel):
    id: str
    label: float


def background_crash():
    time.sleep(35)
    global crash
    crash = True


@app.on_event("startup")
def load_model():
    model_path = PATH_TO_MODEL
    desc_path = PATH_TO_DESCRIPTION
    global model
    model = None
    logger.info('Start to load model')
    time.sleep(20)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    global desc
    logger.info('Model has been loaded!')
    desc = classes.read_attempt_params(desc_path)



@app.get("/")
async def root():
    return {"message": f'You use the model, described as follows: \n {desc}'}


@app.get("/predict/")
def predict(request: DataEntity):
    return make_predict(request.data, request.features, model, desc)


@app.get("/health", status_code=200)
def health(background_task: BackgroundTasks, response: Response) -> bool:
    global counter
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return False
    elif counter < MAX_RESPONSE:
        counter += 1
        msg = f'You just use {counter} of {MAX_RESPONSE} requests'
        logger.info(msg)
        return True
    else:
        logger.error('Service unavailable')
        response.status_code = status.HTTP_404_NOT_FOUND
        return False


def make_predict(data, labels, model, params):
    x_test = pd.DataFrame(data, columns=labels)
    features = prepare_data_to_train(x_test, params)
    y_test = model.predict(features)
    ids = [int(x) for x in range(len(data))]
    return [
        PredictResponce(id=id_, label=float(label)) for id_, label in zip(ids, y_test)
    ]


class custom_transformer():
    def __init__(self, params: classes.AttemptParams):
        self.params = params

    def transform(self, init_data: pd.DataFrame):
        params = self.params
        feature_list = []
        for feature in params.dataset.categorical:
            feature_list.append(process_categorical_data(init_data[feature], params.preprocessing))
        for feature in params.dataset.numerical:
            feature_list.append(init_data[feature])
        if params.preprocessing.binomial:
            for feature in params.dataset.binomial:
                feature_list.append(process_categorical_data(init_data[feature], params.preprocessing))
        else:
            for feature in params.dataset.binomial:
                feature_list.append(init_data[feature])
        prep_feature = pd.concat(feature_list, axis=1)
        for feature in params.dataset.drop_feature:
            prep_feature.drop(columns=feature, inplace=True)
        return prep_feature

    def fit(self, init_data):
        return self


def process_categorical_data(df: pd.Series, params: classes.PreprocessingParams) -> pd.DataFrame:
    enc = None
    df = pd.DataFrame(df)
    if params.categorical == 'OrdinalEncoding':
        enc = OrdinalEncoder()
        enc.fit(df)
        return pd.DataFrame(enc.transform(df))
    elif params.categorical == 'OneHotEncoding':
        enc = OneHotEncoder()
        enc.fit(df)
        return pd.DataFrame(enc.transform(df).toarray())
    return df


def prepare_data_to_train(x_test, params: classes.AttemptParams) -> (pd.DataFrame):
    transformer = custom_transformer(params)
    prep_data = transformer.transform(x_test)
    return prep_data


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))

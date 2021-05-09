import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src import AttemptParams, DatasetParams, PreprocessingParams
from src.data import CAT_FEATURE_PREPROCESSING, read_dataset


def process_categorical_data(df: pd.Series, params: PreprocessingParams) -> pd.DataFrame:
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


def prepare_data_to_train(params: AttemptParams) -> (pd.DataFrame, pd.DataFrame):
    init_data = read_dataset(params.dataset_path)
    feature_list = []
    for feature in params.dataset.categorical:
        feature_list.append(process_categorical_data(init_data[feature],  params.preprocessing))
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
    return prep_feature, pd.DataFrame(init_data[params.dataset.target])


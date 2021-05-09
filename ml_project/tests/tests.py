import pytest
import numpy as np
import pandas as pd
from textwrap import dedent

import ml_project.src.data as data
import ml_project.src.classes as cls
import ml_project.src.features as ftr

SIMPLE_YAML_CAT = 3
SIMPLE_YAML_NUM = 4
SIMPLE_YAML_BIN = 2
SIMPLE_YAML_CONFIG = dedent("""preprocessing:
  categorical: "OneHotEncoding"
  numerical: null
  binomial: False
dataset:
  categorical:
    - "Cat_1"
    - "Cat_2"
    - "Cat_3"
  binomial:
    - "Bin_1"
    - "Bin_2"
  numerical:
    - "Num_1"
    - "Num_2"
    - "Num_3"
    - "Num_4"
  drop_feature:
    - "Num_3"
  target: "Target" 
  test_rate: 0.4
model: "RandomForestClassifier" """)


@pytest.fixture()
def get_simple_params():
    categorical = ["Cat_1", "Cat_2", "Cat_3"]
    numerical = ["Num_1", "Num_2", "Num_3", "Num_4"]
    binomial = ["Bin_1", "Bin_2"]
    target = "Target"
    drop_feature = ["Num_3"]
    test_rate = 0.4
    dataset = cls.DatasetParams(categorical=categorical, numerical=numerical,
                                binomial=binomial, target=target, drop_feature=drop_feature, test_rate=test_rate)

    pre_categorical = "OneHotEncoding"
    pre_numerical = None
    pre_binomial = False
    preprocessing = cls.PreprocessingParams(categorical=pre_categorical, numerical=pre_numerical,
                                            binomial=pre_binomial)
    model = "RandomForestClassifier"
    simple_params = cls.AttemptParams(dataset=dataset, preprocessing=preprocessing, model=model)
    return simple_params


@pytest.fixture()
def get_path_simple_params(get_simple_params, tmp_path):
    simple_config_fio = tmp_path / "simple.yaml"
    simple_config_fio.write_text(SIMPLE_YAML_CONFIG)
    return simple_config_fio


def test_can_generate_num_feature_frame():
    mean = np.random.random(1)[0]
    deviation = np.random.random(1)[0] + 1
    n_samples = np.random.randint(100, 120)
    name = 'num_test'
    data_frame = data.generate_random_numerical(mean, deviation, n_samples, name=name)
    assert (n_samples,) == data_frame.shape, \
        f'DataFrame shape does not correspond to input data, it should be {(n_samples, 1)}, but it appears to be {data_frame.shape}'
    assert name == data_frame.name


def test_can_generate_cat_feature_frame():
    num_cats = np.random.randint(3, 8)
    n_samples = np.random.randint(100, 120)
    probs = np.array([np.random.random() for _ in range(num_cats)])
    probs = probs / np.sum(probs)
    bias = np.random.randint(0, 3)
    name = 'cat_test'
    data_frame = data.generate_random_categorical(num_cats, n_samples, probs, bias, name=name)
    assert (n_samples,) == data_frame.shape, \
        f'DataFrame shape does not correspond to input data, it should be {(n_samples, 1)}, but it appears to be {data_frame.shape}'
    assert name == data_frame.name


def test_can_generate_bin_feature_frame():
    num_cats = 2
    n_samples = np.random.randint(100, 120)
    prob = np.random.random()
    name = 'bin_test'
    data_frame = data.generate_random_categorical(num_cats, n_samples, [prob, 1 - prob], name=name)
    assert (n_samples,) == data_frame.shape, \
        f'DataFrame shape does not correspond to input data, it should be {(n_samples, 1)}, but it appears to be {data_frame.shape}'
    assert name == data_frame.name


def test_can_generate_dataset():
    num_features = np.random.randint(2, 8)
    cat_features = np.random.randint(2, 8)
    bin_features = np.random.randint(2, 8)
    target = 1
    n_samples = np.random.randint(100, 120)
    data_set = data.generate_random_dataset(cat_features, num_features, bin_features, n_samples)
    expected_shape = (n_samples, num_features + cat_features + bin_features + target)
    assert expected_shape == data_set.shape, \
        f'DataSet shape does not correspond to required value it should be {expected_shape}, but it appears to be {data_set.shape}'


def test_can_read_csv(tmp_path):
    num_features = np.random.randint(2, 8)
    cat_features = np.random.randint(2, 8)
    n_samples = np.random.randint(100, 120)
    data_set = data.generate_random_dataset(cat_features, num_features, n_samples)
    dataset_fio = tmp_path / "dataset.csv"
    dataset_fio.write_text(data_set.to_csv(index=False))
    read_dataset = data.read_dataset(dataset_fio)
    assert data_set.shape == read_dataset.shape, f'Loaded dataset does not correspond to saved one'


def test_can_load_config(get_simple_params, get_path_simple_params):
    loaded_params = cls.read_attempt_params(get_path_simple_params)
    assert loaded_params == get_simple_params


def test_can_generate_dataset_for_config(get_simple_params):
    ds_params = get_simple_params.dataset
    n_samples = 10
    dataset = data.generate_random_dataset(cat_features=SIMPLE_YAML_CAT, num_features=SIMPLE_YAML_NUM,
                                           bin_features=SIMPLE_YAML_BIN, n_sample=n_samples)
    feature_list = set().union(*[ds_params.categorical, ds_params.numerical, ds_params.binomial, [ds_params.target]])
    for feature in feature_list:
        assert feature in dataset, f"{feature} no in generated dataset! {list(dataset.columns)}"


def test_can_ohe_process_cat_feature(get_simple_params):
    test_lst = [0, 1, 2, 0, 1, 3]
    test_data = pd.Series(test_lst)
    proc_data = ftr.process_categorical_data(test_data, get_simple_params.preprocessing)
    res = (len(test_lst), max(test_lst) + 1)
    assert res == proc_data.shape, \
        f"Returned df shape is {proc_data.shape} but it should be {res}"


def test_can_ordenc_process_cat_feature(get_simple_params):
    test_lst = [21, 31, 2, 21, 1, 31]
    num_cat = set(test_lst)
    test_data = pd.Series(test_lst)
    params = get_simple_params.preprocessing
    params.categorical = 'OrdinalEncoding'
    proc_data = ftr.process_categorical_data(test_data, params)
    res = (len(test_lst), 1)
    assert res == proc_data.shape, \
        f"Returned df shape is {proc_data.shape} but it should be {res}"
    found_num = int(proc_data[0].max()) + 1
    assert len(num_cat) == found_num, \
        f"There should be {len(num_cat)} categories, but {found_num} were found"


def test_can_build_dataset_to_train_ohe(get_simple_params, tmp_path):
    ds_params = get_simple_params.dataset
    n_samples = 10
    n_opt = 3
    cat_opt = [n_opt for _ in range(SIMPLE_YAML_CAT)]
    dataset = data.generate_random_dataset(cat_features=SIMPLE_YAML_CAT, num_features=SIMPLE_YAML_NUM,
                                           bin_features=SIMPLE_YAML_BIN, n_sample=n_samples, cat_options=cat_opt)
    dataset_fio = tmp_path / "dataset.csv"
    dataset_fio.write_text(dataset.to_csv(index=False))
    get_simple_params.dataset_path = dataset_fio
    prep_data, prep_target = ftr.prepare_data_to_train(get_simple_params)
    num_cnt = SIMPLE_YAML_NUM
    bin_cnt = SIMPLE_YAML_BIN
    cat_cnt = SIMPLE_YAML_CAT * n_opt
    trg_cnt = 1
    for feature in get_simple_params.dataset.drop_feature:
        if "Cat" in feature:
            cat_cnt -= n_opt
        if "Num" in feature:
            num_cnt -= 1
        if "Bin" in feature:
            bin_cnt -= 1
    waited_shape = (n_samples, num_cnt + bin_cnt + cat_cnt)
    assert waited_shape == prep_data.shape, \
        f"Dataset has shape {prep_data.shape}, but it should be {waited_shape}"
    assert (n_samples, trg_cnt) == prep_target.shape, \
        f"Target has shape {prep_target.shape}, but it should be {(n_samples, trg_cnt)}"

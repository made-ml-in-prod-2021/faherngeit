from typing import List, Optional
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PreprocessingParams:
    categorical: str = field(default="OneHotEncoding")
    numerical: str = field(default=None)
    binomial: bool = field(default=False)


@dataclass()
class DatasetParams:
    categorical: List[str]
    numerical: List[str]
    binomial: List[str]
    drop_feature: List[str]
    target: Optional[str]
    test_rate: float = field(default=0.2)


@dataclass()
class AttemptParams:
    preprocessing: PreprocessingParams
    dataset: DatasetParams
    model: str = field(default="RandomForestClassifier")
    dataset_path: str = field(default="data/external/heart.csv")
    model_output_path: str = field(default="models/")


AttemptParams_Schema = class_schema(AttemptParams)


def read_attempt_params(path: str) -> AttemptParams:
    with open(path, "r") as input_stream:
        schema = AttemptParams_Schema()
        return schema.load(yaml.safe_load(input_stream))


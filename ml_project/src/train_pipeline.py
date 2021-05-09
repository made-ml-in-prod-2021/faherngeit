import logging
from sys import stdout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

import classes as cls
import models as mdl
import features as ftr


MAIN_LOGGER_NAME = 'ml_project_logger'
RES_LOGGER_NAME = 'ml_project_train_results'

main_logger = logging.getLogger(MAIN_LOGGER_NAME)
result_logger = logging.getLogger(RES_LOGGER_NAME)


def launch_train_pipeline(config_path: str):
    msg = f"Loading config from {config_path}"
    main_logger.info(msg)
    params = cls.read_attempt_params(config_path)
    main_logger.info("Config loaded successfully!")

    msg = f"Loading dataset from {params.dataset_path}"
    main_logger.info(msg)
    features, target = ftr.prepare_data_to_train(params)
    main_logger.info("Data loaded successfully!")
    msg = f"Splitting data in train and test sets with ratio: {params.dataset.test_rate}"
    main_logger.info(msg)

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=params.dataset.test_rate)
    main_logger.info("Data split successfully!")
    msg = f"{params.model} is going to be trained…"
    main_logger.info(msg)

    model = mdl.train(x_train, y_train, params)
    main_logger.info("Model has trained!")
    train_score = accuracy_score(y_train, model.predict(x_train))
    test_score = accuracy_score(y_test, model.predict(x_test))
    res_msg = f"{params.model} has trained with {train_score} accuracy on train and {test_score} on test data"
    result_logger.info(res_msg)
    main_logger.info(res_msg)

    mdl.save_model(model, params)
    main_logger.info(f"Pretrained model has been save to {params.model_output_path}{params.model}.pkl")


def launch_eval_pipeline(config_path: str):
    msg = f"Loading config from {config_path}"
    main_logger.info(msg)
    params = cls.read_attempt_params(config_path)
    main_logger.info("Config loaded successfully!")
    model = mdl.load_model(params)
    main_logger.info("Model has been loaded!")
    msg = f"Loading dataset from {params.dataset_path}"
    main_logger.info(msg)
    features, target = ftr.prepare_data_to_train(params)
    main_logger.info("Data loaded successfully!")
    pred_trgt = pd.DataFrame(model.predict(features))
    path = 'ml_project/models/'+params.model+'_res.csv'
    pred_trgt.to_csv(path)
    main_logger.info(f"Prediction has been saved to {path}")



def setup_logger():
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    main_handler = logging.StreamHandler(stdout)
    main_handler.setFormatter(formatter)
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(main_handler)
    res_formatter = logging.Formatter('%(asctime)s: %(message)s')
    res_handler = logging.FileHandler('ml_project/models/train_res.log', mode='a')
    res_handler.setFormatter(res_formatter)
    result_logger.setLevel(logging.INFO)
    result_logger.addHandler(res_handler)


def callback_train(args):
    launch_train_pipeline(args.path)


def callback_eval(args):
    launch_eval_pipeline(args.path)


def setup_parser(parser):
    """Setup perser parameters"""
    subparser = parser.add_subparsers(help="Choose command")
    train_parser = subparser.add_parser(
        "train",
        help="Train new model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "-p", '--path',
        help="Path to config file",
        dest="path",
        metavar='PATH',
        required=True,
    )
    train_parser.set_defaults(callback=callback_train)

    eval_parser = subparser.add_parser(
        "predict",
        help="Evaluate model on test set",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "-p", '--path',
        help="Path to config file",
        dest="path",
        metavar='PATH',
        required=True
    )
    eval_parser.set_defaults(callback=callback_eval)


def main():
    parser = ArgumentParser(
        prog='ml_project',
        description='Test application for ML in Prod course',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    setup_logger()
    arguments = parser.parse_args()
    arguments.callback(arguments)

if __name__ == "__main__":
    main()

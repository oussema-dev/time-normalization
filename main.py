"""This is the main module that runs the whole project
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import argparse
import warnings
import os
from utils.processing import (
    create_folders,
    download_gutenburg_db,
    process_data,
    load_data,
    separate_data,
    drop_extra_columns,
)
from utils.normalization import zero_padding, normalize_by_body_mass, interpolate
from machine_learning.data_split import subject_wise_split
from machine_learning.models import initiate_model
from machine_learning.save_metrics_and_plots import save_accuracy_and_plot

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(invalid="ignore")
pd.options.mode.chained_assignment = None
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def main(arguments):
    """This is the main module function

    Parameters:
    arguments (argparse.namespace object): different arguments to be
    passed when running from console

    Returns:
    None
    """

    if arguments.force_data_reload:
        create_folders()
        download_gutenburg_db()
        process_data()
    else:
        ap, ml, v = load_data(arguments.data_type)
        y, subject_id = separate_data(ap)

        if arguments.data_type == "RAW":
            if arguments.normalization_strategy == "zeropadding":
                ap, ml, v = zero_padding(ap, ml, v)
            ap, ml, v = normalize_by_body_mass(ap, ml, v)

        X_data1, X_data2, X_data3 = drop_extra_columns(ap, ml, v)
        if arguments.normalization_strategy == "interpolation":
            X_data1, X_data2, X_data3 = interpolate(X_data1, X_data2, X_data3)

        # Stack the data to make a tensor
        X = np.stack([X_data1.values, X_data2.values, X_data3.values], axis=-1)

        # Splitting the data into train and test sets
        X_train, X_test, y_train, y_test, _, _ = subject_wise_split(
            X, y, subject_id, subject_wise=True, test_size=0.2, random_state=42
        )

        # One hot encode labels
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Initialize an empty list to store the results
        results = []
        print("Training model...")
        for i in range(10):
            print("Iteration", i + 1)
            model = initiate_model(arguments.model_type, X_train)
            y_test_class = np.argmax(y_test, axis=1)

            # Train the model
            history = model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.1,
                verbose=False,
            )

            # Generate predictions on the test set
            y_pred = model.predict(X_test)
            y_pred_class = np.argmax(y_pred, axis=1)

            report = classification_report(y_test_class, y_pred_class, output_dict=True)
            results.append(report["accuracy"])

        # Calculate the mean and standard deviation of the accuracy
        mean_accuracy = sum(results) / len(results)
        std_accuracy = (
            sum((x - mean_accuracy) ** 2 for x in results) / len(results)
        ) ** 0.5

        save_accuracy_and_plot(
            arguments.data_type,
            arguments.normalization_strategy,
            arguments.model_type,
            mean_accuracy,
            std_accuracy,
            history,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force_data_reload",
        action="store_true",
        help="if specified, the data will be reloaded, otherwise, the existing data will be used",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="RAW",
        choices=["RAW", "PRO"],
        help="specify data type (RAW or PRO for processed data)",
    )
    parser.add_argument(
        "--normalization_strategy",
        type=str,
        default="zeropadding",
        choices=["zeropadding", "interpolation"],
        help="specify normalization strategy (zeropadding or interpolation)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "lstm"],
        help="specify model type (cnn or lstm)",
    )

    args = parser.parse_args()
    main(args)
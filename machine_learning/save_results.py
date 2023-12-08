"""This module provides a function to save the model results"""

import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

from config import logs_fld


def save_results(
    data_type,
    normalization_strategy,
    model_type,
    mean_f1,
    std_f1,
    mean_accuracy,
    std_accuracy,
    history,
):
    """this function saves the f1 score metric and the train validation loss plot.

    Parameters:
    data_type (str): data type used to train the model
    normalization_strategy (str): normlization strategy used when loading the data
    model_type (str): model type used for the prediction
    mean_f1 (float): mean f1 score calculated during the prediction
    std_f1 (float): standard deviation of the f1 score
    mean_accuracy (float): mean accuracy calculated during the prediction
    std_accuracy (float): standard deviation of the accuracy
    history (dict): dictionary recording the train and validation loss during the training epochs

    Returns:
    none
    """
    if data_type == "PRO":
        normalization_strategy = "None"
    text = (
        "Data type: "
        + data_type
        + "\nNormalization strategy: "
        + normalization_strategy
        + "\nModel type: "
        + model_type
        + "\nMean f1 score: "
        + str(np.round(mean_f1, 2))
        + "\nf1 score standard deviation: "
        + str(np.round(std_f1, 2))
        + "\nMean accuracy: "
        + str(np.round(mean_accuracy, 2))
        + "\naccuracy standard deviation: "
        + str(np.round(std_accuracy, 2))
    )
    print(text)

    # Create a runtime log folder
    if not os.path.exists(logs_fld):
        os.makedirs(logs_fld)
    now = datetime.datetime.now()
    folder_name = os.path.join(logs_fld, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.mkdir(folder_name)

    filepath = os.path.join(folder_name, "metrics.txt")
    with open(filepath, "w") as f:
        f.write(text)

    # Access training and validation loss from the history object
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    # Create a line plot for training and validation loss
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, "b", label="Training Loss")
    plt.plot(epochs, validation_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(folder_name, "train_validiation_loss.png"))
    plt.close()

    print("Results saved to", folder_name)

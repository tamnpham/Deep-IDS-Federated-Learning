import os

import flwr as fl
from datasets import load_NBaIoT, load_UNSW
from models_ultils import load_ann, load_ann2, load_gru
from datasets import *
import argparse

from sklearn import metrics
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Federated Learning - Client")
parser.add_argument('index', type=int, help='Client Index')

if __name__ == "__main__":

    args = parser.parse_args()

    #! N-BaIoT: 115, 11
    #! BoT-IoT: 13,4
    #! UNSW-NB15: 43,10
    # model = LogisticRegression()

    #! Preprare Dataset
    x_train, x_test, y_train, y_test = load_NBaIoT()

    #! Prepare Model
    model = load_ann2(115, 11)
    # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    # Define Flower client
    class Client(fl.client.NumPyClient):

        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            print("=====================CLIENT INDEX====================")
            print(args.index)
            print("=====================================================")

            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred,axis=1)
            y_eval = np.argmax(y_test,axis=1)
            print(confusion_matrix(y_eval, y_pred))
            print(classification_report(y_eval, y_pred))
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8888", client=Client())

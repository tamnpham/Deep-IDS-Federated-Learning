import flwr as fl
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from tensorflow.keras import models
import os
import numpy as np

history_loss = []
history_acc = []

def get_eval_fn(model, x_test, y_test):
    #! Preprare Dataset
    
    # reshape input to be [samples, time steps, features]
    # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)  # Update model with the latest parameters

        print("=====================================")
        print("Server is evaluating....")
        print("=====================================")

        loss, accuracy = model.evaluate(x_test, y_test)
        history_loss.append(loss)
        history_acc.append(accuracy)
        print(history_loss)
        print(history_acc)
        print("=================TEST REPORT====================")
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred,axis=1)
        y_eval = np.argmax(y_test,axis=1)
        print(confusion_matrix(y_eval, y_pred))
        print(classification_report(y_eval, y_pred))
        print("=================TEST REPORT====================")


        print("==============SAVE CURRENT MODEL=============")
        models.save_model(model, './Storages/fl.NBaIoT-ANN-100r-212.h5')
        print("==============SAVE CURRENT MODEL=============")


        # print("==============PREPARE FOR NEW ROUND=============")
        # os.system('./kill_each_round.sh')
        # os.system('./run_client.sh')
        # print("==============PREPARE FOR NEW ROUND=============")

        return loss, {"accuracy": accuracy}
    return evaluate
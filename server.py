import flwr as fl
from server_ultils import *
from models_ultils import *
from datasets import *


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    
    #? N-BaIoT: 115, 11
    #? BoT-IoT: 13,4
    #? UNSW-NB15: 41,

    x_test, y_test = load_NBaIoT_Test()

    #! Initial Model for training
    model = load_ann2(115,11)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,  # Sample 10% of available clients for the next round
        # min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
        min_available_clients=2,
        eval_fn=get_eval_fn(model, x_test, y_test)
    )
   
    fl.server.start_server("0.0.0.0:8888", config={"num_rounds": 100},strategy=strategy)



from typing import List, Tuple, Callable, Dict, Optional, Tuple, OrderedDict
import argparse

import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import flwr as fl
from flwr.common import Metrics
import utils
from net import Net
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    # executado ao final de cada round
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("olaaa", " acurácia: ", sum(accuracies) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples), "accuracies": (np.array(accuracies)/np.array(examples)).tolist()}

def get_on_fit_config_fn():
    """Return a function which returns training configurations."""

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 2,
        }
        return config

    return fit_config





def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--algorithm",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the algorithm id",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        choices=range(0, 1000),
        required=False,
        help="Specifies the number of rounds",
    )
    args = parser.parse_args()

    # Define strategy
    index = args.algorithm
    ROUNDS = args.rounds
    strategy_name = ["FedAvg", "QFedAvg", "FedAdagrad", "FedYogi", "FedAvgM"][index]

    def get_eval_fn(server_round,
            weight,
            di
    ) -> [float, Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""

        def evaluate(weights) -> Optional[Tuple[float, dict]]:
            """Use the entire CIFAR-10 test set for evaluation."""

            model = Net()
            model.set_weights(weights)
            model.to(DEVICE)

            testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
            loss, accuracy = utils.test(model, testloader, device=DEVICE)
            return loss, {"accuracy": accuracy}

        r = evaluate(weight)
        print("retornou")
        print(r)
        return r


    trainset, testset, num_samples = load_data()
    trainloader, testloader = DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

    model = Net()
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    fedavg = fl.server.strategy.FedAvg(
        # fraction_fit=0.2,
        # fraction_evaluate=0.2,
        # min_fit_clients=2,
        # min_evaluate_clients=2,
        # min_available_clients=10,
        # evaluate_fn=get_evaluate_fn(model, args.toy),
        # on_fit_config_fn=fit_config,
        # on_evaluate_config_fn=evaluate_config,
        evaluate_fn=get_eval_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(),
        #on_evaluate_config_fn=get_on_fit_config_fn(),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))

    qfedavg = fl.server.strategy.QFedAvg(
        q_param=0.6,
        evaluate_fn=get_eval_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(),
        #on_evaluate_config_fn=get_on_fit_config_fn(),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))

    fedadagrad = fl.server.strategy.FedAdagrad(evaluate_metrics_aggregation_fn=weighted_average,
                                               on_fit_config_fn=get_on_fit_config_fn(),
                                               evaluate_fn=get_eval_fn,
                                               #on_evaluate_config_fn=get_on_fit_config_fn(),
                                               initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))

    fedyogi = fl.server.strategy.FedYogi(evaluate_metrics_aggregation_fn=weighted_average,
                                               on_fit_config_fn=get_on_fit_config_fn(),
                                               evaluate_fn=get_eval_fn,
                                               #on_evaluate_config_fn=get_on_fit_config_fn(),
                                               initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))

    fedavgm = fl.server.strategy.FedAvgM(evaluate_metrics_aggregation_fn=weighted_average,
                                               on_fit_config_fn=get_on_fit_config_fn(),
                                               evaluate_fn=get_eval_fn,
                                               #on_evaluate_config_fn=get_on_fit_config_fn(),
                                               initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))

    strategy_dict = {'FedAvg': fedavg, 'QFedAvg': qfedavg, 'FedAdagrad': fedadagrad, 'FedYogi': fedyogi, 'FedAvgM': fedavgm}
    strategy = strategy_dict[strategy_name]
    print("Solução: ", strategy_name)
    # Start Flower server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    losses_distributed = history.losses_distributed
    losses_centralized = history.losses_centralized[1:]
    metrics_distributed = history.metrics_distributed
    metrics_centralized = history.metrics_centralized
    print(history)
    df_losses = pd.DataFrame({'Solution': [strategy_name]*len(losses_distributed),
                                          'Round': [i[0] for i in losses_distributed],
                                          'Loss distributed': [i[1] for i in losses_distributed],
                              'Loss centralized': [i[1] for i in losses_centralized]})

    directory = "output_data1/"
    df_losses.to_csv("""{}{}_loss.csv""".format(directory, strategy_name), index=False)


    print(metrics_distributed)

    # accuracy
    accuracies_distributed = []
    accuracies_centralized = []
    rounds = []
    for i in range(len(metrics_distributed['accuracies'])):

        accuracies_distributed = accuracies_distributed + metrics_distributed['accuracies'][i][1]
        rounds = rounds + [metrics_distributed['accuracies'][i][0]]*len(metrics_distributed['accuracies'][i][1])

    accuracies_centralized = metrics_centralized['accuracy'][1:]
    print("tes", metrics_centralized, accuracies_centralized)
    accuracies_centralized_dict = {i: j for i, j in accuracies_centralized}
    print("Rodada")
    accuracies_centralized = [accuracies_centralized_dict[i] for i in rounds]
    print(accuracies_centralized_dict)
    print(rounds)
    print(accuracies_centralized)

    df_acc = pd.DataFrame({'Solution': [strategy_name] * len(accuracies_distributed),
                                          'Round': rounds,
                                          'Accuracy distributed': accuracies_distributed,
                           'Accuracy centralized': accuracies_centralized})

    df_acc.to_csv("""{}{}_acc.csv""".format(directory, strategy_name), index=False)

    print("-------")
    print(strategy_name)
    print(history)

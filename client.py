import flwr as fl
import torch 
import argparse
from collections import OrderedDict
from centralised import load_data, load_model, train, test
import os

# Server address for connecting to the Flower server.
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")

# Select GPU if available, otherwise use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameters(model, parameters):
    # Load server-provided parameters into the local model.
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({K : torch.tensor(v) for K, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True, help="Client id: 0..num_clients-1")
parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
args = parser.parse_args()

# Create model and load this client's train/test split.
net = load_model()
trainloader, testloader = load_data(partition_id=args.cid, num_partitions=args.num_clients)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Return local model weights to the server.
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        # Train the local model for one round and return updated weights.
        set_parameters(net, parameters)
        train(net, trainloader, device=device, num_epochs=5)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Evaluate the current global model on local test data.
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader, device=device)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    
# Start the Flower client and connect to the server.
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FlowerClient())

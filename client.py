import flwr as fl
import torch 
import argparse
from collections import OrderedDict
from centralised import load_data, load_model, train, test

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({K : torch.tensor(v) for K, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True, help="Client id: 0..num_clients-1")
parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
args = parser.parse_args()

net = load_model()
trainloader, testloader = load_data(partition_id=args.cid, num_partitions=args.num_clients)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, device="cpu", num_epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader, device="cpu")
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(),)

import flwr as fl

def weighted_average(metrics):
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}
    acc = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    return {"accuracy": acc}

strategy = fl.server.strategy.FedAvg(
    min_available_clients=3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
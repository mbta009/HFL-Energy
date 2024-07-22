import yaml
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class EnergyConfig:
    n_flop: float
    p_transmission: float
    h: float
    n_0: float
    b: float
    frequency: float
    sign: float
    c_cpu: float
    c_gpu: float
    utilization: float
    dvfs_gpu: bool
    dvfs_cpu: bool
    gpu: bool

@dataclass
class DataConfig:
    n_samples: int
    dist: list[float]

@dataclass
class PredefinedClientSelectionConfig:
    # edge id -> list of clients
    edge: list[list[int]]
    # round number -> edge id -> list of clients
    rounds: list[list[list[int]]]

class ConfigLoader:

    @staticmethod
    def load_energy_config(path="./config/energy_config.yaml", num_clients=10) -> dict[int, EnergyConfig]:
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        config = dict()
        for i in range(num_clients):
            overrides = config_data.get(f"client_{i}")
            client_config: dict = config_data["default"]
            if overrides:
                client_config.update(overrides)
            config[i] = EnergyConfig(**client_config)
        return config
    
    @staticmethod
    def load_client_selection_config(client_ids: set[int], path="./config/client_selection_config.yaml", num_edges=3, num_rounds=5) -> PredefinedClientSelectionConfig:
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        config = PredefinedClientSelectionConfig(
            edge = [[] for _ in range(num_edges)],
            rounds = [[[] for _ in range(num_edges)] for _ in range(num_rounds)]
        )
        # Fill edges
        client_pool = client_ids.copy()
        empty_edges = set([i for i in range(num_edges)])
        for edge_id, clients in config_data["edge"].items():
            config.edge[edge_id] = clients
            client_pool = client_pool - set(clients)
            empty_edges.remove(edge_id)
        for empty_edge in empty_edges:
            selected_list = random.sample(client_pool, len(client_pool) // len(empty_edges))
            config.edge[empty_edge] = selected_list
            client_pool = client_pool - set(selected_list)
        if len(empty_edges) != 0:
            config.edge[random.sample(empty_edges, 1)[0]].extend(client_pool)
        # Fill rounds
        for round, edges in config_data["rounds"].items():
            for edge_id, clients in edges["edges"].items():
                config.rounds[round][edge_id] = clients
        # We dont fill the rounds, we fill them on fly
        return config
    
    @staticmethod
    def load_data_config(path="./config/data_config.yaml", num_clients=10) -> dict[int, DataConfig]:
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        config = dict()
        for i in range(num_clients):
            overrides = config_data.get(f"client_{i}")
            client_config: dict = config_data["default"]
            if overrides:
                client_config = overrides
            config[i] = DataConfig(**client_config)
        return config
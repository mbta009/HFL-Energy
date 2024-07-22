from abc import ABC, abstractmethod
from typing import Any
import edge
import numpy as np
import config.configloader

class ClientSelectionAlgorithm(ABC):
    @abstractmethod
    def select_edge_server_clients(self, edge_id: int, client_per_edge: int, additional_data: dict[str, Any]) -> list[int]:
        """
        This method will be called when the clients are going to be associated
        to each edge server.
        
        edge_id is the ID of the edge which client are going to be associated with.
        client_per_edge is the number of clients which should be associated with this edge.
        additional_data is simply a dictionary from arbitrary keys to any value which might be needed for client selection.

        The return value must be a list of client IDs which are going to be associated with this edge ID.
        """
        raise Exception("Please implement")
        
    @abstractmethod
    def select_client_for_train(self, edge_id: int, to_select_clients: int, additional_data: dict[str, Any]) -> list[int]:
        """
        This method will be called each time when 
        
        edge_id is the ID of the edge which client are going to be selected.
        to_select_clients is the number of clients to select.
        additional_data is simply a dictionary from arbitrary keys to any value which might be needed for client selection.

        The return value must be a list of client IDs which are going to be selected with this edge.
        """
        raise Exception("Please implement")
    
class ClientSelectionRandom(ClientSelectionAlgorithm):
    def select_edge_server_clients(self, edge_id: int, client_per_edge: int, additional_data: dict[str, Any]) -> list[int]:
        return list(np.random.choice(additional_data["cids"], client_per_edge, replace=False))
        
    def select_client_for_train(self, edge: edge.Edge, to_select_clients: int, additional_data: dict[str, Any]) -> list[int]:
        return list(np.random.choice(edge.cids, to_select_clients, replace = False, p=additional_data["p_clients"]))
    
class ClientSelectionWithPredefinedEdgeClients(ClientSelectionAlgorithm):
    def __init__(self, predefined_clients: dict[int, list[int]]):
        """
        predefined_clients must be a dict from edge id to list of the clients which will be
        associated with the edge
        """
        self.predefined_clients = predefined_clients

    def select_edge_server_clients(self, edge_id: int, client_per_edge: int, additional_data: dict[str, Any]) -> list[int]:
        return self.predefined_clients[edge_id]
        
    def select_client_for_train(self, edge: edge.Edge, to_select_clients: int, additional_data: dict[str, Any]) -> list[int]:
        return list(np.random.choice(edge.cids, to_select_clients, replace = False, p=additional_data["p_clients"]))
    
class PredefinedClientSelection(ClientSelectionAlgorithm):
    def __init__(self, predefined_data: config.configloader.PredefinedClientSelectionConfig):
        self.predefined_data = predefined_data

    def select_edge_server_clients(self, edge_id: int, client_per_edge: int, additional_data: dict[str, Any]) -> list[int]:
        # TODO: what if client_per_edge is not equal to the size of this?
        return self.predefined_data.edge[edge_id]
        
    def select_client_for_train(self, edge: edge.Edge, to_select_clients: int, additional_data: dict[str, Any]) -> list[int]:
        """
        Additional data should contain a round parameter which is a int that says the round number
        """
        selected_clients = self.predefined_data.rounds[int(additional_data["round"])][int(edge.id)]
        if len(selected_clients) == 0: # not defined in config -> random
            return list(np.random.choice(edge.cids, to_select_clients, replace = False))
        return selected_clients
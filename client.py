# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from config.configloader import EnergyConfig
from custom_logging.energy_logging import EnergyLogger
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ProcessUnitType(Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class ProcessUnit:
    process_type: ProcessUnitType
    id: int


class Client:

    def __init__(self, id, train_loader, test_loader, args):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        # record local update epoch
        # record the time
        self.client_processors = []

    def register_processor(self, client_processor):
        self.client_processors.append(client_processor)

    def get_total_energy(self):
        return sum(
            [
                client_processor.client_energy.total_energy
                for client_processor in self.client_processors
            ]
        )

    # def get_config(self):
    #     return self.client_energy.config


class ClientProcessor:

    def __init__(
        self,
        id,
        train_loader,
        test_loader,
        args,
        device,
        processor_type=ProcessUnitType.CPU,
        config=None,
        client=None
    ):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        # record local update epoch
        self.epoch = 0
        # record the time
        # self.process_unit = process_unit
        self.processor_type = processor_type
        self.client: Client = client
        self.client_energy = ClientEnergy(config, self)
        self.clock = []

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(
                    input_batch=inputs, label_batch=labels
                )
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end:
                break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(
            client_id=self.id,
            cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()),
        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

    def get_config(self):
        return self.client_energy.config

    def calculate_energy(self, round) -> float:
        return self.client_energy.calculate_energy(round)

    def __str__(self) -> str:
        return f"Client {self.client.id} - Procees {self.id} - ProcessType {self.processor_type}"

    def __repr__(self) -> str:
        return f"Client {self.client.id} - Procees {self.id} - ProcessType {self.processor_type}"

    def register_client(self, client):
        self.client = client


class ClientEnergy:

    L = 3.49 * 10**5

    def __init__(self, config: EnergyConfig, client_processor: ClientProcessor) -> None:
        self.config = config
        self.client_processor = client_processor
        self.transmission_time = 0
        self.computation_time = 0
        self.e_communication = 0
        self.e_computation = 0
        self.total_energy = 0
        self.logger = EnergyLogger(client_processor)

    def energy(self):
        c = (
            self.config.c_gpu
            if self.client_processor.processor_type == ProcessUnitType.GPU
            else self.config.c_cpu
        )
        workload = len(self.client_processor.train_loader) * self.config.n_flop
        n = (self.config.sign / c) ** (1 / 3)
        f_k = self.config.frequency * n
        self.computation_time = workload / (f_k * self.config.utilization)
        self.e_computation = c * workload * (f_k**2)

    def e_comm(self):
        r = self.config.b * np.log2(
            1 + self.config.p_transmission * self.config.h / self.config.n_0
        )
        self.transmission_time = float(self.L / r)
        self.e_communication = self.config.b * self.config.p_transmission

    def calculate_energy(self, round) -> float:
        self.energy()
        self.e_comm()
        self.total_energy = self.e_computation + self.e_communication
        self.logger.log(self, round)
        return self.total_energy


# class Client():

#     def __init__(self, id, train_loader, test_loader, args, device, config=None):
# =======

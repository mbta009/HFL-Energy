
from __future__ import annotations
import logging
import yaml

from logging import Formatter
from custom_logging.setup_logger import setup_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import ClientProcessor

class YamlFormatter(Formatter):
    def format(self, record):
        log_record = {
            f"round_{record.round}": {
                'time': self.formatTime(record),
                'name': record.name,
                'message': record.msg,
                "result":  record.result
            }
        }
        return yaml.dump(log_record, default_flow_style=False)


class EnergyLogger():
    
    def __init__(self, processor: ClientProcessor, level=logging.INFO) -> None:
        client_id = processor.client.id
        processor_type = processor.processor_type.value
        log_file = f"./logs/client_{client_id}_{processor_type}.yaml"
        self.logger = setup_logger(f"client {client_id} {processor_type}", log_file, YamlFormatter())

    def log(self, energy_client, round):
        total_energy = energy_client.e_communication + energy_client.e_computation
        extra = {
            "result": {
                "transmission_time": energy_client.transmission_time,
                "computation_time": energy_client.computation_time,
                "e_communication": energy_client.e_communication,
                "e_computation": energy_client.e_computation,
                "total_energy": total_energy,
            },
            "round": round
        }
        self.logger.info(f"Result of round = {round}", extra=extra)

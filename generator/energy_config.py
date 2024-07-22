import argparse
import random
import uuid
import yaml

CONFIG_KEYS = ["n_flop", "p_transmission", "h", "n_0", "b", "frequency", "sign", "n"]

def generate_value_for_config_key(key):
    match key:
        case "n_flop":
            return random.random()
        case "p_transmission":
            return random.random()
        case "h":
            return random.random()
        case "n_0":
            return random.random()
        case "b":
            return random.random()
        case "frequency":
            return random.random()
        case "sign":
            return random.random()
        case "n":
            return random.random()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 10,
        help = 'number of all available clients',        
    )
    parser.add_argument(
        '--num_override_clients',
        type = int,
        default = 3,
        help = 'number of all available clients',        
    )
    args = parser.parse_args()
    return args

def get_default_config():
    default_config = dict()
    for key in CONFIG_KEYS:
        default_config[key] = generate_value_for_config_key(key)
    return default_config

if __name__ == "__main__":
    args = parse_args()
    client_numbers = args.num_clients
    number_of_clients_to_be_overrided = args.num_override_clients
    random_sample = random.sample([i for i in range(client_numbers)], number_of_clients_to_be_overrided)
    config = {
        'default': get_default_config()
    }
    for i in random_sample:
        config_key = f"client_{i}"
        client_config = dict()
        override_keys = random.sample(CONFIG_KEYS, random.randint(1, len(CONFIG_KEYS)))
        for key in override_keys:
            client_config[key] = generate_value_for_config_key(key)
        config[config_key] = client_config
    
    with open(f"generator/generated_configs/energy-{uuid.uuid4()}", "w+") as f:
        yaml.dump(config, f, default_flow_style=False)
    
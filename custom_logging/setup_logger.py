import logging
import os

def setup_logger(name, log_file, formatter=None, level=logging.INFO):
    
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    
    with open(log_file, "w"):
        pass

    handler = logging.FileHandler(log_file)
    
    if not formatter:
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

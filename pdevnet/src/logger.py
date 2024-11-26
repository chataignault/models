import os
import logging


def initialise_logger(log_dir, log_file_name, logger_name, log_level):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the full path for the log file
    log_file_path = os.path.join(log_dir, log_file_name)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

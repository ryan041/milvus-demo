import logging
import sys


def create_logger():
    logging.getLogger().setLevel(logging.INFO)

    logger = logging.getLogger("ryan-logger")
    LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    return logger



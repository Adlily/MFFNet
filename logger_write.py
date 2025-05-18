import logging
import os, time, sys

from config import config
save_path = config.SAVE_PATH

def logger_init():

    sys.stderr = sys.stdout

    # path_log = 'log'
    # if os.path.exists(path_log) is False:
        # os.makedirs(path_log)

    # creat a logger and define its class with 'info'
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # create a handler

    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    path = os.path.join(save_path,'log')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, current_time + '.txt')
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)

    # define the output format of the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)


    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
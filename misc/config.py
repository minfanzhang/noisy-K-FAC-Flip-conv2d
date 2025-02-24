import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)

    batch_size_str = str(config.batch_size) + "batch/"
    if config.use_flip :
        use_flip = "Flip/"
    else :
        use_flip = "NoFlip/"

    config.summary_dir = os.path.join("./experiments", config.dataset, config.exp_name, "summary/", batch_size_str, use_flip)
    config.checkpoint_dir = os.path.join("./experiments", config.dataset, config.exp_name, "checkpoint/")
    return config

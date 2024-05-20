import os

import yaml


def read_yaml_file(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def ensure_directory_exists(file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            return True

        return True
    except Exception as message:
        print(f"Impossible to create the CACHE folder: {file_path} - error: {message}")
        return False

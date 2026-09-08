
import pickle
import yaml
import json


def save_artifact(artifact:object, save_path:str) -> None:
    with open(save_path, 'wb') as file:
        pickle.dump(artifact, file, protocol=-1)


def load_artifact(artifact_path:str) -> object:
    with open(artifact_path, 'rb') as file:
        obj = pickle.load(file)

    return obj


def read_yaml(yaml_file_path:str) -> dict:
    with open(yaml_file_path, 'r') as file:
        yaml_file = yaml.safe_load(file)

    return yaml_file


def read_json(filepath:str):
    with open(filepath, 'r') as file:
        j = json.load(file)

    return j


def write_json(data:dict, filepath:str):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)
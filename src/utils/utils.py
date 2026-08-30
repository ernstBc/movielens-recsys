
import pickle
import yaml


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
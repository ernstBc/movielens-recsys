
import pickle


def save_artifact(artifact:object, save_path:str) -> None:
    with open(save_path, 'wb') as file:
        pickle.dump(artifact, file, protocol=-1)


def load_artifact(artifact_path:str) -> object:
    with open(artifact_path, 'rb') as file:
        obj = pickle.load(file)

    return obj
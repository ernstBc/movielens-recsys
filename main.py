from src.train.trainer import ConfigManager
from src.pipeline.pipe import Pipeline
import argparse


parser = argparse.ArgumentParser(description="Pipeline Args")

parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--finetuning", type=bool, default=False, help="Do finetuning.")
parser.add_argument('--model_type', 
                    type=str, 
                    default='DEEP_MATRIX_FACTORIZATION', 
                    choices=['autoencoder', 'deep_autoencoder', 'matrix_factorization', 'deep_matrix_factorization', 'neumf'],
                    help="Model Name ['autoencoder', 'deep_autoencoder', 'matrix_factorization', 'deep_matrix_factorization', 'neumf]")
parser.add_argument('--dataset_type', 
                    type=str, 
                    default='USER_ITEM', 
                    choices=['autoencoder', 'user_item', 'negative_sampling'],
                    help="Dataset Type: ['autoencoder', 'user_item', 'negative_sampling']")
parser.add_argument('--dataset_size', 
                    type=str, 
                    default='100k', 
                    choices=['100k', '10m'],
                    help="MoviesLen subset: ['100k', '10m']")

args = parser.parse_args()



def main():
    N_EPOCHS = args.epochs
    FINETUNING = args.finetuning
    MODEL_TYPE = args.model_type.upper()
    DATASET_TYPE = args.dataset_type.upper()
    DATASIZE = args.dataset_size.upper()
    print(MODEL_TYPE)
    print(DATASET_TYPE)
    print(DATASIZE)

    model_config_path = 'config/models_config.yaml'
    data_dir_config_path = 'config/config.yaml'
    dataset_config_path = 'config/dataset_config.yaml'
    hyperparams_config_path = 'config/hyperparams_config.yaml'
    trainer_config_path = 'config/trainer_config.yaml'
    if FINETUNING:
        finetuning_config_path = 'config/finetuning_config.yaml'

        pipe = Pipeline(
            model_type=MODEL_TYPE,
            dataset_type=DATASET_TYPE,
            data_size=DATASIZE,
            data_dir_config_path=data_dir_config_path,
            dataset_config_path=dataset_config_path,
            model_config_path=model_config_path,
            hyperparams_config_path=hyperparams_config_path,
            trainer_config_path=trainer_config_path,
            finetuning_config_path=finetuning_config_path
        )

    pipe = Pipeline(
            model_type=MODEL_TYPE,
            dataset_type=DATASET_TYPE,
            data_size=DATASIZE,
            data_dir_config_path=data_dir_config_path,
            dataset_config_path=dataset_config_path,
            model_config_path=model_config_path,
            hyperparams_config_path=hyperparams_config_path,
            trainer_config_path=trainer_config_path
        )

    pipe.run_pipeline(max_epochs=N_EPOCHS)



if __name__=='__main__':
    main()
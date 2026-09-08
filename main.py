from src.logger import logging
from src.pipeline.pipe import Pipeline
import argparse
from pytorch_lightning import seed_everything



parser = argparse.ArgumentParser(description="Pipeline Args")

parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument('-f', "--finetuning", action='store_true', help="Do finetuning.")
parser.add_argument('--model_type', 
                    type=str, 
                    required=True,
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
parser.add_argument('--force_process',
                    action='store_true',
                    help="Forces the spliting process every time the dataloader is used.")
parser.add_argument('--process_data',
                    action='store_true',
                    help="Forces the data download process.")
parser.add_argument('--n_epochs_trial', type=int, default=10, help='Number of epochs in each hyperparameter search trial.')
parser.add_argument('--n_trials', type=int, default=10, help='Number of trial in the finetuning process.')
parser.add_argument('--prototype', action='store_true', help="Run prototype pipeline with small dataset,  few epochs and none artifact will be saved locally.")

args = parser.parse_args()


def main():
    N_EPOCHS = args.epochs
    FINETUNING = args.finetuning
    MODEL_TYPE = args.model_type.upper()
    DATASET_TYPE = args.dataset_type.upper()
    DATASIZE = args.dataset_size.upper()
    PROCESS_DATA = args.process_data
    FORCE_DATA = args.force_process
    PROTOTYPE = args.prototype
    N_EPOCHS_TRIAL = args.n_epochs_trial
    N_TRIALS = args.n_trials
    STORAGE = True
    SAVE_MODEL = True


    if PROTOTYPE:
        N_EPOCHS = 1
        FORCE_DATA = False
        PROCESS_DATA = False
        N_EPOCHS_TRIAL = 1
        N_TRIALS = 1
        STORAGE=False
        SAVE_MODEL=False
        

    logging.info('Start Pipeline')

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
            finetuning_config_path=finetuning_config_path,
        )
    else:
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

    pipe.run_pipeline(max_epochs=N_EPOCHS, 
                      max_epochs_finetuning=N_EPOCHS_TRIAL,
                      storage=STORAGE,
                      n_trials=N_TRIALS,
                      save_model=SAVE_MODEL,
                      process_data=PROCESS_DATA,
                      force_process=FORCE_DATA
                      )

    logging.info('End of the Pipeline.')

if __name__=='__main__':
    seed_everything(42)
    main()
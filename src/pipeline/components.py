import os
from src.logger import logging as l
logging = l.getLogger(__name__)


from typing import Literal
from src.train.trainer import ConfigManager
from src.data.wrappers import UserItemDataSampling, AutoencoderSampling
from src.train.trainer import Trainer
from src.fine_tuning.tuner import Tuner
from src.prediction.predict_items import PredictAutoencoderItems, PredictUserAllItems



class DataComponent:
    def __init__(self, data_config:ConfigManager, dataset_config:ConfigManager):
        self.data_config = data_config
        self.dataset_config=dataset_config


    def get_component(self, 
                      dataset_type:str, 
                      dataset_size:str, 
                      process_data:bool=True, 
                      force_process:bool=True, 
                      encode_data:bool=True,
                      data_kwargs:dict={}, 
                      dataset_kwargs:dict={}) -> AutoencoderSampling|UserItemDataSampling:
        if dataset_type.lower() == 'autoencoder':
            args = self.data_config(**data_kwargs) | self.dataset_config(**dataset_kwargs)

            dsize_url = 'SMALL_URL' if dataset_size.lower() == '100k' else 'FULL_URL'
            
            dl_args = {
                 'dataset_url': args['DATA_URL'][dsize_url],
                 'dataset_path': args['DATA_DIR'][dataset_size.upper()]['RAW'],
                 'splits':args['DATA_DIR'][dataset_size.upper()]['SPLITS'],
                 'dataset_name':args['DATA_DIR'][dataset_size.upper()]['NAME'], 
                 'split_mode':args['split_mode'], 
                 'batch_size': args['batch_size'], 
                 'num_workers': args['num_workers'],
                 'process_data':process_data,

            }

            logging.info(f'Arguments for the Autoencoder Data Component: {str(dl_args)} ')
            component = AutoencoderSampling(**dl_args)

        else:
            args = self.data_config(**data_kwargs) | self.dataset_config(**dataset_kwargs)
            

            dsize_url = 'SMALL_URL' if dataset_size.lower() == '100k' else 'FULL_URL'
            dl_args = {
                 'dataset_url': args['DATA_URL'][dsize_url],
                 'dataset_path': args['DATA_DIR'][dataset_size.upper()]['RAW'],
                 'splits':args['DATA_DIR'][dataset_size.upper()]['SPLITS'],
                 'dataset_name':args['DATA_DIR'][dataset_size.upper()]['NAME'], 
                 'train_dataset_path': args['DATA_DIR'][dataset_size.upper()]['TRAIN'],
                 'validation_dataset_path': args['DATA_DIR'][dataset_size.upper()]['EVAL'],
                 'test_dataset_path': args['DATA_DIR'][dataset_size.upper()]['TEST'],

                 'split_mode':args['split_mode'], 
                 'negative_sampling':args['negative_sampling'], 
                 'testing': args['testing'],
                 'batch_size': args['batch_size'], 
                 'num_workers': args['num_workers'],
                 'process_data':process_data,
                 'force_process': force_process,
                 'encode_data': encode_data,
                 'encoder_path': args['ARTIFACTS']['ENCODER_PATH'] if encode_data else None
            }

            logging.info(f'Arguments for the UserItem Data Component: {str((dl_args))}')
            component = UserItemDataSampling(**dl_args)

        return component


class TrainerComponent:
    def __init__(self, model_config:ConfigManager, 
                 hyperparams_config:ConfigManager,
                 trainer_config:ConfigManager):
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config



    def get_component(self, max_epochs:int, model_type:str, trainer_kwargs:dict, model_kwargs:dict={}, hyperparams_config:dict={}) -> Trainer:
        model_args = self.model_config(**model_kwargs)
        hyperparams_args = self.hyperparams_config(**hyperparams_config)
        trainer_args = self.trainer_config(**trainer_kwargs)
        args = model_args | hyperparams_args | trainer_args


        trainer_kargs = {
                 'model_type':model_type,
                 'max_epochs':max_epochs,
                 'model_kwargs':model_args, 
                 'hyperparams_kwargs':hyperparams_args, 
                 'verbose': args['verbose'], 
                 'sanity_check_steps':args['sanity_check_steps'], 
                 'profiler':args['profiler'],
                 'save_intermediate_ckpts':args['save_intermediate_ckpts'],
                 'from_checkpoint': args['from_checkpoint']
        } 

        logging.info(f'Arguments for the Trainer Component: {str(trainer_kargs)}')
        trainer = Trainer(**trainer_kargs)
        return trainer


class FineTuningComponent:
    def __init__(self, 
                 model_config:ConfigManager, 
                 hyperparams_config:ConfigManager, 
                 finetuning_config:ConfigManager, 
                 data_dir_config:ConfigManager,
                 trainer_config:ConfigManager,
                 ):
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config
        self.finetuning_config = finetuning_config
        self.trainer_config = trainer_config
        self.data_dir_config = data_dir_config


    def get_component(self, 
                      study_name:str, 
                      n_trials:int, 
                      model_type:str,  
                      dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                      storage:bool=False,
                      trainer_kwargs:dict={}) -> Tuner:

        tuner_args = {
            'study_name':study_name,
            'n_trials':n_trials,
            'model_type':model_type,
            'dataset_type':dataset_type,
            'fine_tuning_config':self.finetuning_config(),
            'data_dir_config':self.data_dir_config(),
            'model_kwargs':self.model_config(),
            'hyperparams_kwargs':self.hyperparams_config(),
            'trainer_kwargs':self.trainer_config(**trainer_kwargs),
            'storage':storage
        }

        logging.info(f'Arguments for the Finetuning Component: {str(tuner_args)} ')
        tuner = Tuner(**tuner_args)

        return tuner


class PredictionComponent:
    def __init__(self, data_config:ConfigManager) -> None:
        self.data_config = data_config


    def get_component(self, dataset_type:str, dataset_size:str, padding:bool=True) -> PredictUserAllItems| PredictAutoencoderItems:
        if dataset_type.lower() == 'autoencoder':
            data_path = os.path.join(
                            self.data_config()['DATA_DIR'][dataset_size.upper()]['RAW'],
                            'ratings.csv')
            movies_df_path = self.data_config()['DATA_DIR'][dataset_size.upper()]['MOVIES_PATH']

            predict_args = {
                'data_path':data_path, 
                'movies_df_path':movies_df_path
            }
            logging.info(f'Arguments for the Autoencoder Predictor: {str(predict_args)}')
            predictor = PredictAutoencoderItems(**predict_args)

        else:
            n_movies = self.data_config()['DATA_DIR'][dataset_size.upper()]['N_MOVIES']
            if padding:
                n_movies +=1

            movies_encoder = self.data_config()['ARTIFACTS']['ENCODER_PATH']
            movies_df_path = self.data_config()['DATA_DIR'][dataset_size.upper()]['MOVIES_PATH']

            predict_args = {
                'n_movies':n_movies, 
                'movies_encoder':movies_encoder,
                'movies_df_path':movies_df_path
            }

            logging.info(f'Arguments for the UserItem Predictor: {str(predict_args)}')
            predictor = PredictUserAllItems(**predict_args)

        return predictor
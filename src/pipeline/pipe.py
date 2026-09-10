import os
from src.logger import logging as l
logging = l.getLogger(__name__)

from src.train.trainer import ConfigManager
from src.pipeline.components import (DataComponent, TrainerComponent, FineTuningComponent, PredictionComponent)

class Pipeline:
    def __init__(self, 
                 model_type:str,
                 dataset_type:str,
                 data_size:str,
                 data_dir_config_path:str,
                 dataset_config_path:str,
                 model_config_path:str,
                 hyperparams_config_path:str,
                 trainer_config_path:str,
                 finetuning_config_path:str|None=None,
                 ):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.datasize = data_size
        self.data_dir_config = ConfigManager(data_dir_config_path)
        self.dataset_config  = ConfigManager(dataset_config_path, config_settings=dataset_type+'_DATASET_CONFIG')
        self.model_config = ConfigManager(model_config_path, config_settings=model_type+'_CONFIG')
        self.hyperparams_config = ConfigManager(hyperparams_config_path)
        self.trainer_config = ConfigManager(trainer_config_path)
        self.finetuning_config = None

        if finetuning_config_path is not None:
            self.finetuning_config = ConfigManager(finetuning_config_path)


    def run_pipeline(self,
                     max_epochs:int = 5,
                     model_kwargs:dict={}, 
                     hyperparams_kwargs:dict={},
                     dataset_kwargs:dict={},
                     data_dir_kwargs:dict={},
                     trainer_kwargs:dict={},
                     max_epochs_finetuning:int=2,
                     n_trials:int=5,
                     save_model:bool=True,
                     force_process=False,
                     process_data=False,
                     storage=False):

        logging.info('Creating a Pipeline.')


        # set pipeline components
        # set empty handler that will be replaced in case that finetuning stage happens
        model_extra_config, hyperparams_extra_config, dataset_extra_config= {}, {}, {}

        # it will inin the training process, it changes to false if the finetuning can 
        # find a better set of hyperparams
        continue_training = True


        # Load DataLoader
        
        dc = DataComponent(data_config=self.data_dir_config,
                           dataset_config=self.dataset_config)
        logging.info('Creating a DataLoader Component')
        dataloader = dc.get_component(
                              dataset_type=self.dataset_type,
                              dataset_size=self.datasize,
                              data_kwargs=data_dir_kwargs,
                              dataset_kwargs=dataset_kwargs,
                              force_process=force_process,
                              process_data=process_data,)

        # Finetuning component
        # Finetune search the best set of hyperparams for a given type of model
        if self.finetuning_config is not None:
            logging.info('Starting the finetuning Process.')
            ftc = FineTuningComponent(
                self.model_config,
                hyperparams_config=self.hyperparams_config,
                finetuning_config=self.finetuning_config,
                trainer_config=self.trainer_config,
                data_dir_config=self.data_dir_config
            )
            tuner = ftc.get_component(
                study_name=self.model_type,
                n_trials=n_trials,
                model_type=self.model_type,
                dataset_type=self.dataset_type, # type: ignore
                trainer_kwargs={'verbose':False, 
                                "sanity_check_steps":0, 
                                'save_intermediate_ckpts':False,
                                },
                storage=storage
            )
            # The hparams will replace the default and user defined parameters
            logging.info('Starting the hyperparameters search')
            tuner.search_params(dataloader=dataloader, max_epochs=max_epochs_finetuning)
            model_extra_config, hyperparams_extra_config, dataset_extra_config = tuner.get_best_params()
            best_metrics = tuner.get_best_metrics()
            candidate_params = {**best_metrics,
                               'model_config': model_extra_config,
                               'hyperparams_config': hyperparams_extra_config,
                               'dataset_config': dataset_extra_config,
                               }
            continue_training = tuner.check_and_save_best_params_(candidate_best_params=candidate_params)
            logging.info(f"The finetuning process has found better hyperparameters: {continue_training}")

            
        if not continue_training:
            logging.warning(f"The Skip Training")
            print('Skip training...')
            return None
        

        # Update the parameters with the new values obtained from the tuner
        model_args = self.model_config(**model_kwargs) | model_extra_config
        hyper_args = self.hyperparams_config(**hyperparams_kwargs) | hyperparams_extra_config
        dataset_args = self.dataset_config() | dataset_extra_config
        trainer_args  = self.trainer_config(**trainer_kwargs) | {"max_epochs": max_epochs}


        logging.info('Creating the Trainer Component')
        tc = TrainerComponent(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            trainer_config=self.trainer_config,
        )
        logging.info('Getting Trainer Component')
        trainer = tc.get_component(
            max_epochs=max_epochs,
            model_type=self.model_type,
            trainer_kwargs=trainer_args,
            hyperparams_config=hyper_args,
            model_kwargs=model_args,
        )


        # Predictor predicts the items and saves the items with the higher values into a csv
        logging.info('Creating the Predictor Component')
        pc = PredictionComponent(data_config=self.data_dir_config)
        logging.info('Getting Predictor Component')
        predictor = pc.get_component(dataset_type=self.dataset_type,
                                     dataset_size=self.datasize)


        # run components
        # dataloader
        dataloader.batch_size = dataset_args['batch_size']

        # trainer
        logging.info('Starting Training Process.')
        model = trainer.get_model()
        training_results = trainer.train(model=model, dataloader=dataloader)
        self.training_results = training_results
        logging.info(f"Training Results: {training_results}")

        if save_model:
            logging.info('Saving the model')
            path = self.data_dir_config()['REGISTRY']['models']
            trainer.save_model(model=model, save_model_path=path)
            logging.info(f'Model saved at {path}')


        # predictor
        logging.info('Starting Prediction Process.')

        save_path = self.data_dir_config()['PREDICTIONS'][self.datasize.upper()]['PATH']
        n_users = self.data_dir_config()['DATA_DIR'][self.datasize.upper()]['N_USERS']
        predictor.predict_all_and_save(model=model, n_users=n_users, save_path=save_path)
        logging.info(f'Predictions saved at {save_path}')
   

        print('Pipeline Process Completed.')
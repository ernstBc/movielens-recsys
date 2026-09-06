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
                     finetuning_kwargs:dict={},
                     max_epochs_finetuning:int=2,
                     n_trials:int=50):


        # set pipeline components
        # set empty handler that will be replaced in case that finetuning stage happens
        model_extra_config, hyperparams_extra_config, dataset_extra_config= {}, {}, {}



        # Load DataLoader
        dc = DataComponent(data_config=self.data_dir_config,
                           dataset_config=self.dataset_config)
        dataloader = dc.get_component(
                              dataset_type=self.dataset_type,
                              dataset_size=self.datasize,
                              data_kwargs=data_dir_kwargs,
                              dataset_kwargs=dataset_kwargs)

        # Finetuning component
        # Finetune search the best set of hyperparams for a given type of model
        if self.finetuning_config is not None:
            ftc = FineTuningComponent(
                self.model_config,
                hyperparams_config=self.hyperparams_config,
                finetuning_config=self.finetuning_config,
                trainer_config=self.trainer_config
            )
            tuner = ftc.get_component(
                study_name=self.model_type,
                n_trials=n_trials,
                model_type=self.model_type,
                dataset_type=self.dataset_type, # type: ignore
                trainer_kwargs=trainer_kwargs ,
                storage=None,
            )
            # 
            tuner.search_params(dataloader=dataloader, max_epochs=max_epochs_finetuning)
            model_extra_config, hyperparams_extra_config, dataset_extra_config = tuner.get_best_params()
            

        # Update the parameters with the new values obtained from the tuner
        model_args = self.model_config(**model_kwargs) | model_extra_config
        hyper_args = self.hyperparams_config(**hyperparams_kwargs) | hyperparams_extra_config
        dataset_args = self.dataset_config() | dataset_extra_config
        trainer_args  = self.trainer_config () | {"max_epochs": max_epochs}

        

        tc = TrainerComponent(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            trainer_config=self.trainer_config,
        )
        trainer = tc.get_component(
            model_type=self.model_type,
            trainer_kwargs=trainer_args,
            hyperparams_config=hyper_args,
            model_kwargs=model_args,
        )


        # Predictor predicts the items and saves the items with the higher values into a csv
        pc = PredictionComponent(data_config=self.data_dir_config)
        predictor = pc.get_component(dataset_type=self.dataset_type,
                                     dataset_size=self.datasize)


        # run components
        # dataloader
        dataloader.batch_size = dataset_args['batch_size']

        # trainer
        model = trainer.get_model()
        training_results = trainer.train(model=model, dataloader=dataloader)

        # predictor
        predictions = predictor.predict(model=model, user_id=12)
        predictions_df = predictor.get_predicted_items(predictions=predictions)

        predictor.save_predictions(predictions_df, 'artifacts/predictions_12.csv')


        self.training_results = training_results

        print('Pipeline Process Completed.')
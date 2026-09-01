import optuna
from src.train.trainer import Trainer, ConfigManager
from typing import Literal

class Tuner:
    def __init__(self, 
                 study_name:str, 
                 n_trials:int,
                 model_type:str,
                 dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                 finetuning_config:ConfigManager,
                 model_config:ConfigManager,
                 hyperparams_config:ConfigManager,
                 data_dir_config: ConfigManager,
                 dataset_config:ConfigManager,
                 storage:str|None = None,
                 pruner:bool=False):


        self.model_type = model_type
        self.finetuning_config = finetuning_config()
        self.n_trials = n_trials
        self.trainer = Trainer(
            model_type=model_type,
            dataset_type=dataset_type,
            model_config=model_config,
            hyperparams_config=hyperparams_config,
            data_dir_config=data_dir_config,
            dataset_config=dataset_config
        )

        if storage is not None:
            if pruner:
                self.study = optuna.create_study(study_name=study_name, 
                                                 storage=storage, 
                                                 load_if_exists=True,
                                                 pruner=optuna.pruners.MedianPruner())
            else:
                self.study = optuna.create_study(study_name=study_name, 
                                                 storage=storage, 
                                                 load_if_exists=True)
        else:
            if pruner:
                self.study = optuna.create_study(study_name=study_name,
                                                 load_if_exists=True,
                                                 pruner=optuna.pruners.MedianPruner())
            self.study = optuna.create_study(study_name=study_name, load_if_exists=True)


    def _objective(self, trial, max_epochs:int):
        # Model Params Grid
        model_params = {}
        for param_name, param_config in self.finetuning_config[self.model_type.upper()].items():
            if param_config['type'] == 'categorical':
                model_params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                model_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                model_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config['logscale'] )
            elif param_config['type'] == 'list_int':
                model_params[param_name] = [trial.suggest_int(f"{param_name}_{i+1}", param_config['low'], param_config['high'], step=param_config['step'])
                                                 for i in range(
                                                                trial.suggest_int(f"{param_name}_nlist", 
                                                                param_config['min_elements'], 
                                                                param_config['max_elements'])
                                                                )
                                            ]
            else:
                raise ValueError(f"Unsupported parameter type: ;{param_config['type']};")

        # Hypermeters Grid
        hyper_params = {}
        for param_name, param_config in self.finetuning_config['HYPERPARAMS'].items():
            if param_config['type'] == 'categorical':
                hyper_params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                hyper_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config['step'])
            elif param_config['type'] == 'float':
                hyper_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'list_int':
                hyper_params[param_name] = [trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config['step'])
                                             for _ in range(trial.suggest_int(f"{param_name}_nlist", param_config['min_elements']), param_config['max_elements'])]
            else:
                raise ValueError(f"Unsupported parameter type: '{param_config['type']}'")

        # Dataset hyperparams Grid
        dataset_params = {}
        for param_name, param_config in self.finetuning_config['DATASET'].items():
            if param_config['type'] == 'categorical':
                dataset_params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                dataset_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config['step'])
            elif param_config['type'] == 'float':
                dataset_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'list_int':
                dataset_params[param_name] = [trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config['step'])
                                             for _ in range(trial.suggest_int(f"{param_name}_nlist", param_config['min_elements']), param_config['max_elements'])]
            else:
                raise ValueError(f"Unsupported parameter type: '{param_config['type']}'")

        # Reset the trainer params 
        self.trainer.reset_params('model_config', model_params)
        self.trainer.reset_params('hyperparams_config', hyper_params)
        self.trainer.reset_params('dataset_config', dataset_params)
        self.trainer.setup()

        # train the model with new hyperparams
        self.trainer.train(max_epochs=max_epochs, save_model=False, verbose=False)

        # get the results
        results = self.trainer.evaluate(verbose=False)[0]['test_loss_epoch']

        return results


    def search_params(self, max_epochs:int):
        self.study.optimize(
            lambda trial: self._objective(trial, max_epochs), 
            n_trials=self.n_trials, 
            show_progress_bar=True)
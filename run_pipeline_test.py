from src.train.trainer import ConfigManager
from src.pipeline.components import DataPipeline


if __name__ == '__main__':
    MODEL_TYPE = 'AUTOENCODER'
    DATASET_TYPE = 'USER_ITEM'
    DATASIZE = '100K'

    model_config_path = 'config/models_config.yaml'
    data_dir_config_path = 'config/config.yaml'
    finetuning_config_path = 'config/finetuning_config.yaml'
    dataset_config_path = 'config/dataset_config.yaml'
    hyperparams_config_path = 'config/hyperparams_config.yaml'

    model_config = ConfigManager(model_config_path, config_settings=MODEL_TYPE+'_CONFIG')
    data_dir_config = ConfigManager(data_dir_config_path)
    finetuning_config = ConfigManager(finetuning_config_path, MODEL_TYPE)
    dataset_config = ConfigManager(dataset_config_path, config_settings=DATASET_TYPE+'_DATASET_CONFIG')
    hyperparams_config = ConfigManager(hyperparams_config_path)

    data_pipeline = DataPipeline(data_dir_config, dataset_config)
    dl = data_pipeline.get_component(dataset_type=DATASET_TYPE,
                                     dataset_size=DATASIZE,
                                     process_data=False,
                                     force_process=False)
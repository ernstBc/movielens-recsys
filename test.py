from src.train.trainer import Trainer, ConfigManager
from src.fine_tuning.tuner import Tuner
from src.prediction.predict_items import Predictor
from typing import Literal


def test_model(model_type: Literal['DEEP_MATRIX_FACTORIZATION'], dataset_type: Literal['USER_ITEM'], datasize: Literal['100K']):
    # Load model configuration
    model_config = ConfigManager(model_config_path, config_settings=model_type+'_CONFIG')
    
    # Load dataset configuration
    dataset_config = ConfigManager(dataset_config_path, config_settings=dataset_type+'_DATASET_CONFIG')
    
    # Load hyperparameters configuration
    hyperparams_config = ConfigManager(hyperparams_config_path)

    # load data dir configuration
    data_dir_config = ConfigManager(data_dir_config_path)

    # Load finetuning configuration
    finetuning_config = ConfigManager(finetuning_config_path)

    # Initialize the trainer
    trainer = Trainer(model_type, 
                      dataset_type.lower(), 
                      model_config=model_config, 
                      hyperparams_config=hyperparams_config,
                      data_dir_config=data_dir_config,
                      dataset_config=dataset_config,)
    
    # Train the model
    trainer.train(max_epochs=2)
    print('Trainer Process check.')
    
    # Fine-tune the model
    tuner = Tuner('test', 2, model_type, dataset_type.lower(), finetuning_config, model_config,
                  hyperparams_config, data_dir_config, dataset_config)
    tuner.search_params(max_epochs=1)
    print('Finetuner process Check')
    
    # Predict items for a user
    
    p = Predictor(data_dir_config, model_config)
    predictor = p.get_predictor(model_type=MODEL_TYPE)
    predictions = predictor.predict(trainer.model, user_id=12)  # Example user_id
    preds_df = predictor.get_predicted_items(predictions)
    print('Prediction Component Check')

    return predictions, preds_df


if __name__ =='__main__':

    MODEL_TYPE = 'DEEP_MATRIX_FACTORIZATION'
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



    preds, preds_df = test_model(model_type=MODEL_TYPE, dataset_type=DATASET_TYPE, datasize=DATASET_TYPE)

    print(preds)
    print(preds_df)


    models_type = ['AUTOENCODER', 'DEEP_AUTOENCODER', 'MATRIX_FACTORIZATION', 'DEEP_MATRIX_FACTORIZATION']
    dataset_types = ['AUTOENCODER'] * 2 + ['USER_ITEM']*2


    for MODEL_TYPE, DATASET_TYPE in zip(models_type, dataset_types):
        print('Model type:', MODEL_TYPE)
        print('DATASET_TYPE:', DATASET_TYPE)
        model_config = ConfigManager(model_config_path, config_settings=MODEL_TYPE+'_CONFIG')
        data_dir_config = ConfigManager(data_dir_config_path)
        finetuning_config = ConfigManager(finetuning_config_path, MODEL_TYPE)
        dataset_config = ConfigManager(dataset_config_path, config_settings=DATASET_TYPE+'_DATASET_CONFIG')
        hyperparams_config = ConfigManager(hyperparams_config_path)

        preds, preds_df = test_model(model_type=MODEL_TYPE, dataset_type=DATASET_TYPE, datasize=DATASET_TYPE)

        print(preds)
        print(preds_df)
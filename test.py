from src.train.trainer import Trainer, ConfigManager
from src.fine_tuning.tuner import Tuner
from typing import Literal
from src.pipeline.components import DataComponent, TrainerComponent, FineTuningComponent, PredictionComponent


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


def test_model_v2(
        model_type,
        dataset_type,
        datasize):
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

    # load trainer configuration
    trainer_config = ConfigManager(trainer_config_path)

    # components
    print('init data component')
    dc = DataComponent(data_config=data_dir_config, dataset_config=dataset_config)
    dl = dc.get_component(dataset_type=dataset_type, dataset_size=datasize, process_data=False, force_process=False)
    print(dl)

    print('init trainer component')
    tc = TrainerComponent(model_config=model_config, hyperparams_config=hyperparams_config, trainer_config=trainer_config)
    trainer = tc.get_component(model_type=model_type, trainer_kwargs={'max_epochs':1})
    print(trainer)

    print('init finetuning component')
    ftc = FineTuningComponent(model_config=model_config, hyperparams_config=hyperparams_config, finetuning_config=finetuning_config, trainer_config=trainer_config)
    tuner = ftc.get_component(study_name='a', 
                              n_trials=1, 
                              model_type=model_type, 
                              dataset_type=dataset_type, 
                              storage=None, 
                              trainer_kwargs={'max_epochs':1, 
                                              'save_intermediate_ckpts':False,
                                              'sanity_check_steps':0,
                                              'verbose':False})
    print(tuner)

    print('init predictor component')
    pc = PredictionComponent(data_config=data_dir_config)
    predictor = pc.get_component(dataset_type=dataset_type, dataset_size=datasize)
    print(predictor)

    print('All components works as expected')

    model = trainer.get_model()
    train_results = trainer.train(model, dl)
    print('Train Results:', train_results)

    tuner.search_params(dataloader=dl, max_epochs=1)

    predicts = predictor.predict(model, 12)
    print('----predicts-------', predicts)
    predicts_df = predictor.get_predicted_items(predictions=predicts, top_k=5)
    print('-------predicts df ----------', predicts_df)



    return predicts, predicts_df


if __name__ =='__main__':

    MODEL_TYPE = 'DEEP_AUTOENCODER'
    DATASET_TYPE = 'AUTOENCODER'
    DATASIZE = '100K'

    model_config_path = 'config/models_config.yaml'
    data_dir_config_path = 'config/config.yaml'
    finetuning_config_path = 'config/finetuning_config.yaml'
    dataset_config_path = 'config/dataset_config.yaml'
    hyperparams_config_path = 'config/hyperparams_config.yaml'
    trainer_config_path = 'config/trainer_config.yaml'


    model_config = ConfigManager(model_config_path, config_settings=MODEL_TYPE+'_CONFIG')
    data_dir_config = ConfigManager(data_dir_config_path)
    finetuning_config = ConfigManager(finetuning_config_path, MODEL_TYPE)
    dataset_config = ConfigManager(dataset_config_path, config_settings=DATASET_TYPE+'_DATASET_CONFIG')
    hyperparams_config = ConfigManager(hyperparams_config_path)



    preds, preds_df = test_model_v2(model_type=MODEL_TYPE, dataset_type=DATASET_TYPE, datasize=DATASIZE)

    print(preds)
    print(preds_df)


    models_type = ['AUTOENCODER', 'DEEP_AUTOENCODER', 'MATRIX_FACTORIZATION', 'DEEP_MATRIX_FACTORIZATION']
    dataset_types = ['AUTOENCODER'] * 2 + ['USER_ITEM']*2



    for MODEL_TYPE, DATASET_TYPE in zip(models_type, dataset_types):
        if ('a' =='aa'):
            print('Model type:', MODEL_TYPE)
            print('DATASET_TYPE:', DATASET_TYPE)
            preds, preds_df = test_model_v2(model_type=MODEL_TYPE, dataset_type=DATASET_TYPE, datasize=DATASIZE)

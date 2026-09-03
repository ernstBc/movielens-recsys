import os
import torch
import pandas as pd
import numpy as np
from src.utils.utils import load_artifact
from torch.utils.data import TensorDataset, DataLoader


class PredictUserAllItems:
    def __init__(self, n_movies, movies_encoder, movies_df):
        self.range_movies = torch.arange(n_movies)
        self.movies_encoder = movies_encoder
        self.movies_df = movies_df


    def predict(self, model, user_id):
        user_id_tensor = torch.full_like(self.range_movies, user_id)

        loader = DataLoader(
                    TensorDataset(user_id_tensor, self.range_movies), 
                    batch_size=512, 
                    shuffle=False)

        predictions = []
        with torch.inference_mode():
            for user_batch, movie_batch in loader:
                batch = [user_batch, movie_batch]
                user_predictions = model(batch)

                predictions.extend(user_predictions.tolist())

        return torch.tensor(predictions)


    def get_predicted_items(self, predictions, top_k:int=10):
        values, indices = torch.topk(predictions, k=top_k)

        indices = indices.tolist()
        decoded_indices = [self.movies_encoder.decode_id(decoded_idx) for decoded_idx in indices]

        user_data = pd.DataFrame({"movieId": decoded_indices, "scores": values.tolist()})
        user_data = user_data.merge(self.movies_df, on='movieId', how='inner')

        return user_data


    def save_predictions(self, predictions, save_path:str):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)

        predictions.to_csv(save_path)


class PredictAutoencoderItems:
    def __init__(self, data, movies_df):
        self.data = data
        self.movies_df = movies_df

    def predict(self, model, user_id):
        user_ratings = torch.tensor(self.data.loc[user_id].to_numpy(), dtype=torch.float)

        with torch.inference_mode():
            user_preds = model(user_ratings)

        return user_preds
    

    def get_predicted_items(self, predictions, top_k:int=10):
        values, indices = torch.topk(predictions, k=top_k)

        indices = indices.tolist()
        
        user_data = pd.DataFrame({"movieId": indices, "scores": values.tolist()})
        user_data = user_data.merge(self.movies_df, on='movieId', how='inner')

        return user_data


    def save_predictions(self, predictions, save_path:str):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)

        predictions.to_csv(save_path)



class Predictor:
    def __init__(self, data_dir_config, model_config):
        self.data_dir_config = data_dir_config()
        self.model_config = model_config()


    def get_predictor(self, model_type:str):
        movies_path = self.data_dir_config['DATA_DIR']['100K']['RAW'] + '/movies.csv'
        movies_df = pd.read_csv(movies_path)

        if model_type.lower() in ['autoencoder', 'deep_autoencoder']:
            data_dir = os.path.join(self.data_dir_config['DATA_DIR']['100K']['RAW'], 'ratings.csv')
            data = pd.read_csv(data_dir)
            data = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
            predictor = PredictAutoencoderItems(data, movies_df)
        else:
            n_movies = self.model_config['n_items']
            artifact_path = self.data_dir_config['ARTIFACTS']['ENCODER_PATH']
            movies_encoder = load_artifact(artifact_path)
            predictor = PredictUserAllItems(n_movies, movies_encoder, movies_df)

        return predictor


    


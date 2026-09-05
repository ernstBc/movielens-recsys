import os
import torch
import pandas as pd
import numpy as np
from src.utils.utils import load_artifact
from torch.utils.data import TensorDataset, DataLoader
from src.data.processing_data import MovieIDEncoder


class PredictUserAllItems:
    def __init__(self, n_movies:int, movies_encoder:str|MovieIDEncoder, movies_df_path:str):
        self.range_movies = torch.arange(n_movies)
        self.movies_df = pd.read_csv(movies_df_path)

        if isinstance(movies_encoder, str):
            encoder = load_artifact(movies_encoder)
        elif isinstance(movies_encoder, MovieIDEncoder):
            encoder = movies_encoder
        else:
            raise ValueError('movies_encoder should be a string or a instance of MovieIDEncoder')
        assert isinstance(encoder, MovieIDEncoder), 'encoder must be a instance of MovieIDEncoder'
        self.encoder = encoder


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
        decoded_indices = [self.encoder.decode_id(decoded_idx) for decoded_idx in indices]

        user_data = pd.DataFrame({"movieId": decoded_indices, "scores": values.tolist()})
        user_data = user_data.merge(self.movies_df, on='movieId', how='inner')

        return user_data


    def save_predictions(self, predictions, save_path:str):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)

        predictions.to_csv(save_path)


class PredictAutoencoderItems:
    def __init__(self, data_path:str, movies_df_path:str):
        self.data = self._process_data(pd.read_csv(data_path))
        self.movies_df = pd.read_csv(movies_df_path)


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


    def _process_data(self, data:pd.DataFrame):
        d = data.pivot(index = 'userId', columns='movieId', values='rating').fillna(0)
        return d
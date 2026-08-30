import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader



class PredictUserAllItems:
    def __init__(self, model, n_movies, movies_encoder, movies_df):
        self.model = model
        self.range_movies = torch.arange(n_movies + 1)
        self.movies_encoder = movies_encoder
        self.movies_df = movies_df


    def predict(self, user_id):
        user_id_tensor = torch.full_like(self.range_movies, user_id)

        loader = DataLoader(
                    TensorDataset(user_id_tensor, self.range_movies), 
                    batch_size=256, 
                    shuffle=False)

        predictions = []
        with torch.inference_mode():
            for user_batch, movie_batch in loader:
                user_predictions = self.model(user_batch, movie_batch)

                predictions.extend(user_predictions.tolist())

        return predictions


    def get_predicted_items(self, predictions, top_k:int=10):
        indices, values = torch.topk(predictions, k=top_k)

        indices = indices.tolist()
        decoded_indices = [self.movies_encoder.decode(decoded_idx) for decoded_idx in indices]

        user_data = pd.DataFrame([decoded_indices, values], columns=['movieId', 'predicted_rating'])
        user_data = user_data.merge(self.movies_df, on='movieId', how='inner')

        return user_data
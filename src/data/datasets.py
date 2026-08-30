import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class SimpleDataSet(Dataset):
    def __init__(self, data_path:str) -> None:
        self.data = pd.read_csv(data_path)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]

        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']

        user_id = torch.tensor(user_id, dtype=torch.long)
        movie_id = torch.tensor(movie_id, dtype=torch.long)
        rating = torch.tensor(rating, dtype=torch.float32)

        return user_id, movie_id, rating



class AutoEncoderDataset(Dataset):
    def __init__(self, data_path:str):
        data = pd.read_csv(data_path)
        self.data = self._process_data(data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        ratings_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        return ratings_tensor


    def _process_data(self, data:pd.DataFrame) -> np.ndarray:
        pivot_data = data.pivot(columns='movieId', index='userId', values='rating')
        pivot_data = pivot_data.fillna(0)
        pivot_data = pivot_data.to_numpy()
        return pivot_data


class NegSampleDataset(Dataset):
    def __init__(self, data_path:str, num_negatives:int=4):
        self.data = pd.read_csv(data_path)
        self.num_negatives = num_negatives
        self.user_movie_set = set(zip(self.data['userId'], self.data['movieId']))
        self.all_movies = self.data['movieId'].unique()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        user_id = self.data.iloc[idx]['userId']
        movie_id = self.data.iloc[idx]['movieId']
        rating = self.data.iloc[idx]['rating']

        # Create a list of negative samples
        negative_samples = []
        while len(negative_samples) < self.num_negatives:
            neg_movie_id = torch.randint(0, len(self.all_movies), (1,)).item()
            if (user_id, neg_movie_id) not in self.user_movie_set:
                negative_samples.append(neg_movie_id)

        user_id_tensor = torch.tensor(user_id, dtype=torch.long)
        movie_id_tensor = torch.tensor(movie_id, dtype=torch.long)
        rating_tensor = torch.tensor(rating, dtype=torch.float32)
        negative_samples_tensor = torch.tensor(negative_samples, dtype=torch.long)

        return user_id_tensor, movie_id_tensor, rating_tensor, negative_samples_tensor
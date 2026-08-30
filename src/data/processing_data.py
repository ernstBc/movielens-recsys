import os
import random
import pandas as pd 
from typing import List, Literal, Tuple



class ProcessData:
    def __init__(self, 
                 raw_data_path:str, 
                 destination_paths:List[str], 
                 splits:List[float],
                 mode:Literal['random', 'user', 'user_time'] = 'random',
                 encode_data:bool = False):
        """
        Processing Data Step.
        It split the data into two or three different sets.
            Args:
                raw_data_path: Directory of the raw dataset.
                destination_path: Directory where to save the processed data.
                splits: A list of the percentages of each split. [train, validation] or [train, validation, test]
                mode [random, user, user_time]: How the data is split.
        
        """

        assert len(destination_paths) == len(splits), f"Destination Path must have the same number of elements than the splits, but got {len(destination_paths)} vs {len(splits)}"
        assert len(splits) in [2, 3], 'The splits list needs 2 or 3 values'
        assert sum(splits) == 1.0, 'The sum of the splits list must sum up 1.0'
        assert mode in ['random', 'user', 'user_time'], "mode argument only takes ['random', 'user', 'user_time'] as possible values."

        self.raw_data_path = raw_data_path
        self.mode = mode
        self.splits = splits

        paths = {'train_path': destination_paths[0], 'val_path': destination_paths[1]}
        if len(destination_paths) == 3:
            paths['test_path'] = destination_paths[2]

        self.destination_paths = paths
        self.test_split = True if len(destination_paths) == 3 else False
        self.encode_data = encode_data


    def process_data(self, force_process=False) -> None:
        data_in_destination = self.check_data()

        if (data_in_destination) and (force_process is False):
            print('Data already in local path.')
        else:
            df = pd.read_csv(self.raw_data_path)

            if self.encode_data:
                encoder = MovieIDEncoder()
                encoder.fit(df['movieId'].to_list())
                self.encoder = encoder
                df['movieId'] = df['movieId'].apply(encoder.encode_id)

            if self.mode == 'random':
                dfs = self._process_random(df)
            elif self.mode =='user':
                dfs = self._process_per_user(df)
            else:
                dfs = self._process_per_user(df, time_aware=True)

            self._save_data(dfs)


    def _process_random(self, df:pd.DataFrame) -> Tuple[pd.Series, ...]:
        """
        Split the data randomly into two or three sets.
        
        """
        n_elements = df.shape[0]
        indices = list(range(n_elements))
        random.shuffle(indices)

        train_size, val_size = int(n_elements * self.splits[0]), int(n_elements * self.splits[1])
        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size+val_size)]

        train_df = df.iloc[train_indices, :]
        val_df = df.iloc[val_indices, :]

        if self.test_split:
            test_size = int(n_elements * self.splits[2])
            test_indices = indices[-test_size:]
            test_df = df.iloc[test_indices, :]

            return (train_df, val_df, test_df)

        return (train_df, val_df)


    def _process_per_user(self, df:pd.DataFrame, time_aware:bool=False) -> Tuple[pd.DataFrame, ...]:
        if time_aware:
            # order the data chronologically
            df = df.sort_values(by = ['userId', 'timestamp'], ascending=[True,True])
        else:
            # shuffle data
            df = df.sample(frac=1.0)

        df_columns = df.columns
        df_grouped = df.groupby('userId').agg(list)
        train_data = []
        val_data = []
        test_data = []

        for row in df_grouped.iterrows(): 
            # row[0] = user idx, row[1] = pd.Series with movieId, ratings, timestamp as rows
            user_id = row[0]
            values = row[1].values

            len_row = len(values[0])

            train_size = int(self.splits[0] * len_row)
            val_size = int(self.splits[1] * len_row) 
            test_size = 0

            train_user_samples = []
            val_user_sample = []
            test_user_sample = []
            
            if self.test_split:
                test_size = int(self.splits[2] * len_row) 
            
            # iter training samples
            run_values = [v[:train_size] for v in values] # training values
            for movie_id, rating, timestamp in zip(*run_values):
                example = (user_id, movie_id, rating, timestamp)
                train_user_samples.append(example)

            # iter validation samples
            run_values = [v[train_size: (train_size+val_size)] for v in values] # training values
            for movie_id, rating, timestamp in zip(*run_values):
                example = (user_id, movie_id, rating, timestamp)
                val_user_sample.append(example)

            # iter test samples
            if self.test_split:
                run_values = [v[-test_size:] for v in values] # training values
                for movie_id, rating, timestamp in zip(*run_values):
                    example = (user_id, movie_id, rating, timestamp)
                    test_user_sample.append(example)

            train_data.extend(train_user_samples)
            val_data.extend(val_user_sample)
            test_data.extend(test_user_sample)

        # convert the data into dataframes
        train_df = pd.DataFrame(train_data, columns=df_columns)
        val_df = pd.DataFrame(val_data, columns=df_columns)
        if self.test_split:
            test_df = pd.DataFrame(test_data, columns=df_columns)

            return train_df, val_df, test_df
        
        return train_df, val_df


    def _save_data(self, dfs: Tuple[pd.DataFrame]) -> None:
        """Saves the datasets.
            Args:
                dfs: Tuple of pandas DataFrame. The number of dataframes must match the length 
                     of destination paths."""
        for df, (split_name, path_destination) in zip(dfs, self.destination_paths.items()):
            print(f"Saving {split_name} to: {path_destination}")
            df.to_csv(path_destination)


    def get_encoder(self):
        if self.encode_data:
            return self.encoder
        else:
            return None


    def check_data(self):
        """Check if the data is already processed at the destination path."""
        if any([os.path.exists(path) for path in self.destination_paths]):
            return True
        else:
            return False



class MovieIDEncoder:
    def __init__(self):
        self.encoded_ids = {}
        self.decoded_ids = {}
        

    def fit(self, data:list) -> None:
        for idx, d in enumerate(data, 1):
            self.encoded_ids[d] = idx
            self.decoded_ids[idx] = d


    def encode_id(self, idx:int) -> int:
        return self.encoded_ids[idx]


    def decode_id(self, idx:int) -> int:
        return self.decoded_ids[idx]





if __name__ == '__main__':
    pass



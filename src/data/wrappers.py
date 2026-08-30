import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.datasets import AutoEncoderDataset, NegSampleDataset, SimpleDataSet
from src.data.get_data import GetData
from src.data.processing_data import ProcessData
from typing import Literal


class AutoencoderSampling(pl.LightningDataModule):
    def __init__(self, data_config:dict, data_size:Literal['100k', '10m']):
        super().__init__()
        self.data_config = data_config
        self.data_size = f"DATA_{data_size.upper()}"


    def prepare_data(self) -> None:
        url_set = 'SMALL_URL' if self.data_size == 'DATA_100K' else 'FULL_URL'
        dataset_url = self.data_config['DATA_URL'][url_set]
        dataset_path = self.data_config['DATA_DIR'][self.data_size]['RAW']
        dataset_name = self.data_config['DATA_DIR'][self.data_size]['NAME']

        gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
        gt.get_data()


    def setup(self, stage: str) -> None:
        data_config_path = os.path.join(self.data_config['DATA_DIR'][self.data_size]['RAW'], 'ratings.csv')
        splits = self.data_config['DATA_DIR'][self.data_size]['SPLITS']
        full_dataset = AutoEncoderDataset(data_path=data_config_path)

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, splits)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=False)


class UserItemDataSampling(pl.LightningDataModule):
    def __init__(self, 
                 config:dict, 
                 split_mode:Literal['random', 'user', 'user_time'], 
                 data_size:Literal['100k', '100m'], 
                 negative_sampling:bool=False, 
                 testing:bool=True):
        super().__init__()

        self.data_config = config
        self.negative_sampling = negative_sampling
        self.testing = testing
        self.data_size = f"DATA_{data_size.upper()}"
        self.split_mode = split_mode


    def prepare_data(self) -> None:
        url_set = 'SMALL_URL' if self.data_size == 'DATA_100K' else 'FULL_URL'
        dataset_url = self.data_config['DATA_URL'][url_set]
        dataset_path = self.data_config['DATA_DIR'][self.data_size]['RAW']
        dataset_name = self.data_config['DATA_DIR'][self.data_size]['NAME']

        gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
        gt.get_data()

        splits = self.data_config['DATA_DIR'][self.data_size]['SPLITS']
        train_set_path = self.data_config['DATA_DIR'][self.data_size]['TRAIN']
        val_set_path = self.data_config['DATA_DIR'][self.data_size]['EVAL']
        test_set_path = self.data_config['DATA_DIR'][self.data_size]['TEST']
        destination_paths = [train_set_path, val_set_path, test_set_path]

        if (len(splits)==3) and (self.testing is False):
            splits = [splits[0], splits[1] + splits[2]]
            destination_paths.pop()

        pd = ProcessData(raw_data_path=dataset_path,
                         destination_paths=destination_paths, 
                         splits=splits, 
                         mode=self.split_mode)

        pd.process_data(force_process=True)


    def setup(self, stage: str) -> None:
        train_set_path = self.data_config['DATA_DIR'][self.data_size]['TRAIN']
        val_set_path = self.data_config['DATA_DIR'][self.data_size]['EVAL']

        if self.negative_sampling:
            train_dataset = NegSampleDataset(data_path=train_set_path)
            val_dataset = NegSampleDataset(data_path=val_set_path)
        else:
            train_dataset = SimpleDataSet(data_path=train_set_path)
            val_dataset = SimpleDataSet(data_path=val_set_path)

        if self.testing:
            test_set_path = self.data_config['DATA_DIR'][self.data_size]['TEST']

            if self.negative_sampling:
                test_dataset = NegSampleDataset(data_path=test_set_path)
            else:
                test_dataset = SimpleDataSet(data_path=test_set_path)

            self.test_dataset = test_dataset

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.testing:
            return DataLoader(self.test_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=False)
        else:
            return DataLoader(self.val_dataset, batch_size=self.data_config['BATCH_SIZE'], shuffle=False)
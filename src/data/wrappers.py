import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.datasets import AutoEncoderDataset, NegSampleDataset, UserItemDataset
from src.data.get_data import GetData
from src.data.processing_data import ProcessData
from typing import Literal
from src.train.customs import negative_sampling_collate_fn


class AutoencoderSampling(pl.LightningDataModule):
    def __init__(self, data_dir_config:dict, dataset_config:dict):
        super().__init__()
        self.data_dir_config = data_dir_config
        self.dataset_config = dataset_config


    def prepare_data(self) -> None:
        data_size = self.dataset_config['data_size']
        url_set = 'SMALL_URL' if data_size == '100K' else 'FULL_URL'

        dataset_url = self.data_dir_config['DATA_URL'][url_set]
        dataset_path = self.data_dir_config['DATA_DIR'][data_size]['RAW']
        dataset_name = self.data_dir_config['DATA_DIR'][data_size]['NAME']

        gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
        gt.get_data()


    def setup(self, stage: str) -> None:
        data_size = self.dataset_config['data_size']
        data_config_path = os.path.join(self.data_dir_config['DATA_DIR'][data_size]['RAW'], 'ratings.csv')
        splits = self.data_dir_config['DATA_DIR'][data_size]['SPLITS']
        full_dataset = AutoEncoderDataset(data_path=data_config_path)

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, splits)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.dataset_config['batch_size'], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.dataset_config['batch_size'], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.dataset_config['batch_size'], shuffle=False)


class UserItemDataSampling(pl.LightningDataModule):
    def __init__(self, data_dir_config:dict, dataset_config:dict, force_process:bool=False):
        super().__init__()

        self.data_dir_config = data_dir_config
        self.dataset_config = dataset_config
        self.force_process = force_process


    def prepare_data(self) -> None:
        data_size = self.dataset_config['data_size']
        url_set = 'SMALL_URL' if data_size == '100K' else 'FULL_URL'
        split_mode = self.dataset_config['split_mode']
        testing = self.dataset_config['testing']

        dataset_url = self.data_dir_config['DATA_URL'][url_set]
        dataset_path = self.data_dir_config['DATA_DIR'][data_size]['RAW']
        dataset_name = self.data_dir_config['DATA_DIR'][data_size]['NAME']

        gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
        gt.get_data()

        splits = self.data_dir_config['DATA_DIR'][data_size]['SPLITS']
        train_set_path = self.data_dir_config['DATA_DIR'][data_size]['TRAIN']
        val_set_path = self.data_dir_config['DATA_DIR'][data_size]['EVAL']
        test_set_path = self.data_dir_config['DATA_DIR'][data_size]['TEST']
        destination_paths = [train_set_path, val_set_path, test_set_path]

        if (len(splits)==3) and (testing is False):
            splits = [splits[0], splits[1] + splits[2]]
            destination_paths.pop()

        pd = ProcessData(raw_data_path=dataset_path,
                         destination_paths=destination_paths, 
                         splits=splits, 
                         mode=split_mode)

        pd.process_data(force_process=self.force_process)


    def setup(self, stage: str) -> None:
        data_size = self.dataset_config['data_size']
        negative_sampling = self.dataset_config['negative_sampling']
        testing = self.dataset_config['testing']

        train_set_path = self.data_dir_config['DATA_DIR'][data_size]['TRAIN']
        val_set_path = self.data_dir_config['DATA_DIR'][data_size]['EVAL']

        if negative_sampling:
            num_negatives = self.dataset_config['num_negatives']
            train_dataset = NegSampleDataset(data_path=train_set_path, num_negatives=num_negatives)

        else:
            train_dataset = UserItemDataset(data_path=train_set_path)

        val_dataset = UserItemDataset(data_path=val_set_path)
        if testing:
            test_set_path = self.data_dir_config['DATA_DIR'][data_size]['TEST']

            test_dataset = UserItemDataset(data_path=test_set_path)

            self.test_dataset = test_dataset

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_size=self.dataset_config['batch_size'], 
                          shuffle=True,
                          num_workers=self.dataset_config['num_workers'],
                          collate_fn=negative_sampling_collate_fn if isinstance(self.train_dataset, NegSampleDataset) else None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, 
                          batch_size=self.dataset_config['batch_size'], 
                          shuffle=False,
                          num_workers=self.dataset_config['num_workers'],
                          collate_fn=negative_sampling_collate_fn if isinstance(self.val_dataset, NegSampleDataset) else None)

    def test_dataloader(self) -> DataLoader:
        if self.dataset_config['testing']:
            return DataLoader(self.test_dataset, 
                              batch_size=self.dataset_config['batch_size'], 
                              shuffle=False,
                              num_workers=self.dataset_config['num_workers'],
                              collate_fn=negative_sampling_collate_fn if isinstance(self.test_dataset, NegSampleDataset) else None)
        else:
            return DataLoader(self.val_dataset, 
                              batch_size=self.dataset_config['batch_size'], 
                              shuffle=False,
                              num_workers=self.dataset_config['num_workers'],
                              collate_fn=negative_sampling_collate_fn if isinstance(self.val_dataset, NegSampleDataset) else None)
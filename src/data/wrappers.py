import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.datasets import AutoEncoderDataset, NegSampleDataset, UserItemDataset
from src.data.get_data import GetData
from src.data.processing_data import ProcessData
from typing import Literal
from src.train.customs import negative_sampling_collate_fn


class AutoencoderSampling(pl.LightningDataModule):
    def __init__(self,   
                 dataset_url:str,
                 dataset_path:str,
                 splits:list,
                 dataset_name:str='ml-latest.zip', 
                 split_mode:str='user', 
                 negative_sampling:bool=False, 
                 testing:bool=True,
                 batch_size:int=512, 
                 num_workers:int=1,
                 process_data:bool=True):
        super().__init__()
        self.dataset_url = dataset_url
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.splits = splits

        self.split_mode = split_mode
        self.negative_sampling = negative_sampling
        self.testing = testing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.process_data = process_data



    def prepare_data(self) -> None:
        if self.process_data:
            dataset_url = self.dataset_url
            dataset_path = self.dataset_path
            dataset_name = self.dataset_name

            gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
            gt.get_data()


    def setup(self, stage: str) -> None:
        data_config_path = os.path.join(self.dataset_path, 'ratings.csv')
        splits = self.splits
        full_dataset = AutoEncoderDataset(data_path=data_config_path)

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, splits)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class UserItemDataSampling(pl.LightningDataModule):
    def __init__(self,   
                 dataset_url:str,
                 dataset_path:str,
                 splits:list, 
                 train_dataset_path:str,
                 validation_dataset_path:str,
                 test_dataset_path:str|None=None,
                 dataset_name:str='ml-latest.zip',
                 split_mode:Literal['random', 'user', 'user_time']='user',
                 negative_sampling:bool=False, 
                 testing:bool=True,
                 batch_size:int=512, 
                 num_workers:int=1,
                 num_negatives:int=0,
                 force_process=False,
                 process_data:bool=True):
        super().__init__()

        self.dataset_url = dataset_url
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.splits = splits
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.test_dataset_path = test_dataset_path

        self.split_mode = split_mode
        self.negative_sampling = negative_sampling
        self.testing = testing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_negatives = num_negatives
        self.force_process = force_process
        self.process_data = process_data


        assert split_mode in ['random', 'user', 'user_time'], "split mode must be one of the possible values ['random', 'user', 'user_time']"


    def prepare_data(self) -> None:
        if self.process_data:
            split_mode = Literal[self.split_mode]
            testing = self.testing

            dataset_url = self.dataset_url
            dataset_path = self.dataset_path
            dataset_name = self.dataset_name

            gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
            gt.get_data()

            splits = self.splits
            train_set_path = self.train_dataset_path
            val_set_path = self.validation_dataset_path
            test_set_path = self.test_dataset_path
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
        negative_sampling = self.negative_sampling
        testing = self.testing

        train_set_path = self.train_dataset_path
        val_set_path = self.validation_dataset_path

        if negative_sampling:
            num_negatives = self.num_negatives
            train_dataset = NegSampleDataset(data_path=train_set_path, num_negatives=num_negatives)

        else:
            train_dataset = UserItemDataset(data_path=train_set_path)

        val_dataset = UserItemDataset(data_path=val_set_path)

        if testing and self.test_dataset_path is not None:
            test_set_path = self.test_dataset_path
            test_dataset = UserItemDataset(data_path=test_set_path)
            self.test_dataset = test_dataset

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=negative_sampling_collate_fn if isinstance(self.train_dataset, NegSampleDataset) else None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=negative_sampling_collate_fn if isinstance(self.val_dataset, NegSampleDataset) else None)

    def test_dataloader(self) -> DataLoader:
        if self.testing:
            return DataLoader(self.test_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=False,
                              num_workers=self.num_workers,
                              collate_fn=negative_sampling_collate_fn if isinstance(self.test_dataset, NegSampleDataset) else None)
        else:
            return DataLoader(self.val_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=False,
                              num_workers=self.num_workers,
                              collate_fn=negative_sampling_collate_fn if isinstance(self.val_dataset, NegSampleDataset) else None)
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.datasets import AutoEncoderDataset, NegSampleDataset, SimpleDataSet
from src.data.get_data import GetData
from src.data.processing_data import ProcessData
from src.utils.utils import read_yaml



class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, data_config:dict):
        super().__init__()
        self.data_config = data_config


    def prepare_data(self) -> None:
        dataset_url = self.data_config['DATA_URL']['SMALL_URL']
        dataset_path = self.data_config['DATA_DIR']['DATA_100K']['RAW']
        dataset_name = self.data_config['DATA_DIR']['DATA_100K']['NAME']

        gt = GetData(dataset_url=dataset_url, dataset_path=dataset_path, dataset_name=dataset_name)
        gt.get_data()


    def setup(self, stage: str) -> None:
        data_config_path = os.path.join(self.data_config['DATA_DIR']['DATA_100K']['RAW'], 'ratings.csv')
        splits = self.data_config['DATA_DIR']['DATA_100K']['SPLITS']
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

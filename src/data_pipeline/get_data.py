import os
import zipfile
from urllib import request
import tempfile
from pathlib import Path
import shutil


class GetData:
    def __init__(self, dataset_url: str, dataset_path: str, dataset_name:str):
        self.dataset_url = dataset_url
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name


    def download_dataset(self, file_name):
        """
        Downloads the dataset from the url.
        """
        print('Downloading data...')
        file_name, file_status = request.urlretrieve(self.dataset_url, file_name)
        print('data downloaded')
        return file_name, file_status



    def extract_dataset(self, data_location, data_extracted_location):
        """
        Extracts the downloaded dataset if it is in a zip format.
        """
        print('Extracting data...')
        if data_location.endswith('.zip'):
            with zipfile.ZipFile(data_location, 'r') as zip_ref:
                zip_ref.extractall(data_extracted_location)
        print('data extracted.')


    def get_data(self):
        """
        Download and Extract dataset from URL
        """
        download_needed = self.check()
        if download_needed:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # define tmp file location
                tmp_file_name = os.path.join(tmp_dir, self.dataset_name)

                # download and extract data into tmp dir
                file_name, file_status = self.download_dataset(tmp_file_name)
                self.extract_dataset(file_name, tmp_dir)

                # move the files from tmp dir to the persistent dir
                path_tmpdir = Path(tmp_dir)
                print(os.listdir(path_tmpdir))
                folder_in_tmp_dir = [f.name for f in path_tmpdir.iterdir() if f.is_dir()]
                print('folder in tmp dir:', folder_in_tmp_dir)

                if len(folder_in_tmp_dir) == 0:
                    raise Exception('No folder found in tmp dir')
                
                file_in_tmp_dir_path = os.path.join(tmp_dir, folder_in_tmp_dir[0])
                list_files = [file for file in os.listdir(file_in_tmp_dir_path) if file.endswith('.csv') ]

                print('Moving files ...')
                for file in list_files:
                    print(f'Moving file: {file} to {self.dataset_path}')
                    shutil.move(os.path.join(file_in_tmp_dir_path, file), self.dataset_path)

                print('files moved.')
        else:
            print('Dataset already downloaded and extracted.')


    def check(self):
        """
        Check if the dataset is already downloaded and extracted.
        """
        download_needed = False
        if not os.path.exists(self.dataset_path):
            download_needed = True

        return download_needed


    
if __name__ =='__main__':
    import yaml
    config_yaml = 'config/config.yaml'

    config_file = yaml
import requests
import zipfile
from pathlib import Path
from enum import Enum, auto
import shutil
import os

class DataType(Enum):
    RAW = auto()
    INTERIM = auto()
    PROCESSED = auto()
    RESULTS = auto()

class DataManager:
    def __init__(self, base_dir='./data/', rewrite=False, conf={}):
        self.base_dir = Path(base_dir)
        self.dir_paths = {
            DataType.RAW: Path(conf.get(DataType.RAW, base_dir)) / 'raw',
            DataType.INTERIM: Path(conf.get(DataType.INTERIM, base_dir)) / 'interim',
            DataType.PROCESSED: Path(conf.get(DataType.PROCESSED, base_dir)) / 'processed',
            DataType.RESULTS: Path(conf.get(DataType.RESULTS, base_dir))  / 'results'
        }

        for key in self.dir_paths.keys():
            self.dir_paths[key].mkdir(parents=True, exist_ok=True)
        
        if self.dir_paths[DataType.INTERIM].stat().st_size == 0:
             if not list(self.dir_paths[DataType.INTERIM].iterdir()):
                 self.extract_raw_data()

    def get_path(self, folder_id):
        return self.dir_paths[folder_id]

    def get_wesad(self):
        url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

        response = requests.get(url)
        file_Path = self.dir_paths[DataType.RAW] / 'WESAD.zip'

        if response.status_code == 200:
            with open(file_Path, 'wb') as file:
                file.write(response.content)
            print('File downloaded successfully')
        else:
            print('Failed to download file')

    def extract_raw_data(self):
        if not list(self.dir_paths[DataType.RAW].iterdir()):
            self.get_wesad()

        with zipfile.ZipFile(self.dir_paths[DataType.RAW] / 'WESAD.zip', 'r') as zip_ref:
            pkl_files = [name for name in zip_ref.namelist() if name.endswith('.pkl')]
            
            extracted_files = []
            
            for file_path in pkl_files:
                file_info = zip_ref.getinfo(file_path)
                original_filename = file_info.filename
                file_info.filename = os.path.basename(original_filename)
                
                zip_ref.extract(file_info, self.dir_paths[DataType.INTERIM])
                
                extracted_files.append(file_info.filename)
                print(f"The {file_info.filename} subject was extracted.")
        
        return extracted_files

    def clean_gen_data(self, *args, clean_all=False, confirmation=True):
        dir_to_delete = []

        if clean_all:
            dir_to_delete.append(self.base_dir)
        else:
            for dir in args:
                dir_to_delete.append(self.dir_paths[dir])

        total_size = 0
        file_count = 0
        for dir in dir_to_delete:
            if not Path(dir).exists():
                print(f"There is no folder with {dir} path")
                continue

            for file_path in self.base_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
                
        print(f"To be deleted: {file_count} files, {total_size / 1024 / 1024:.2f} MB")

        if confirmation:
            confirm = input(f"Are you sure you want to clean the following folders: {'  '.join(map(str, dir_to_delete))}? (y/n): ")
            if confirm.lower() != 'y':
                print('The operation was canceled.')
                return 

        for dir in dir_to_delete:                
            shutil.rmtree(Path(dir))

        print("The data was deleted successfully.")

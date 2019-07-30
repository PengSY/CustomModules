import os

from ruamel import yaml
from azure.storage.blob import BlockBlobService


class BlobDownloader:
    META_FILE = '_meta.yaml'

    def __init__(self, account_name, account_key, container):
        self.blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
        self.container = container

    def download_files_from_blob(self, blob_paths, prefix_len, output_folder):
        files = []
        for blob_path in blob_paths:
            file_name = blob_path[prefix_len:]

            folder = os.path.dirname(os.path.join(output_folder, file_name))
            os.makedirs(folder, exist_ok=True)

            print(f"Start downloading file: {file_name}")
            self.blob_service.get_blob_to_path(
                container_name=self.container,
                blob_name=blob_path,
                file_path=os.path.join(output_folder, file_name),
            )
            print(f"End downloading file: {file_name}")
            files.append(file_name)
        print(f"{len(files)} files downloaded")

    def import_folder(self, data_folder, output_folder, folder_type='GenericFolder'):
        if len(data_folder) == 0 or data_folder[-1] != '/':
            data_folder += '/'
        blob_paths = self.blob_service.list_blob_names(self.container, prefix=data_folder)
        self.download_files_from_blob(blob_paths, len(data_folder), output_folder)
        self.ensure_meta(output_folder, folder_type)

    def ensure_meta(self, output_folder, folder_type):
        meta_file_path = os.path.join(output_folder, self.META_FILE)
        if os.path.exists(meta_file_path):
            self._check_meta(meta_file_path, folder_type)
        else:
            self._generate_meta(meta_file_path, folder_type)

    @staticmethod
    def _check_meta(meta_file_path, folder_type):
        try:
            with open(meta_file_path) as fin:
                data = yaml.safe_load(fin)
            if data['type'] != folder_type:
                raise Exception(f"Invalid folder_type in meta, expected: {folder_type}, got {data['type']}")
        except BaseException as e:
            raise Exception(f"Invalid meta file") from e

    @staticmethod
    def _generate_meta(meta_file_path, folder_type):
        data = {'type': folder_type}
        with open(meta_file_path, 'w') as fout:
            yaml.round_trip_dump(data, fout)

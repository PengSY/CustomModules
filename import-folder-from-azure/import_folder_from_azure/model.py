import fire

from .common import BlobDownloader


def import_model_folder(account_name, account_key, container, data_folder, output_folder):
    downloader = BlobDownloader(account_name=account_name, account_key=account_key, container=container)
    downloader.import_folder(data_folder, output_folder, 'ModelFolder')


if __name__ == '__main__':
    fire.Fire(import_model_folder)

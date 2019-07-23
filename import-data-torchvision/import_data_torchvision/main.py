import os

import fire
import ruamel.yaml
from torchvision import datasets


def write_meta(dataset_name, yaml_path):
    data = {'type': 'GenericFolder', 'dataset-type': 'torchvision-dataset', 'dataset-name': dataset_name}
    with open(yaml_path, 'w') as fout:
        ruamel.yaml.round_trip_dump(data, fout)


DATASET_NAMES = {
    'CIFAR10', 'CIFAR100', 'Caltech101', 'Caltech256', 'CelebA', 'FashionMNIST',
    'ImageNet', 'KMNIST', 'MNIST', 'Omniglot', 'PhotoTour',
    'SBDataset', 'SBU', 'SEMEION', 'STL10', 'SVHN', 'VOCDetection', 'VOCSegmentation',
}


def download_from_dataset(dataset_name, output_folder):
    if dataset_name not in DATASET_NAMES:
        raise Exception(f"Not a valid dataset name: {dataset_name}")
    load_data_method = getattr(datasets, dataset_name)
    print(f"Start downloading dataset {dataset_name}")
    ds = load_data_method(output_folder, download=True)
    print(f"Dataset downloaded: {ds}")
    return ds


def download_dataset(dataset_name, output_folder):
    download_from_dataset(dataset_name, output_folder)
    meta_file = '_meta.yaml'
    write_meta(dataset_name, os.path.join(output_folder, meta_file))
    print(f"Meta data file '{meta_file}'")


if __name__ == '__main__':
    fire.Fire(download_dataset)

from setuptools import setup


setup(
    name="azureml-custom-module-import-data-torchvision",
    version="0.0.3",
    description="A custom module for importing torchvision datasets.",
    packages=["import_data_torchvision"],
    author="Heyi Tang",
    license="MIT",
    include_package_data=True,
)

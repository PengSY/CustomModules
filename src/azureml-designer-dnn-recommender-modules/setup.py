from azureml.designer.modules.recommenders.dnn.package_info import PACKAGE_NAME, VERSION

__import__('sys').path.insert(0, '..')
perform_setup = getattr(__import__('setup_template'), 'perform_setup')

if __name__ == '__main__':
    perform_setup(
        package_name=PACKAGE_NAME,
        version=VERSION,
        data_files=('*.yaml',),
    )

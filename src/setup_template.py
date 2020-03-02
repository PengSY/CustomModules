import os
import re

from setuptools import setup, find_packages

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


def get_requirements():
    exclude = ('pytest', 'pylint')
    install_reqs = parse_requirements('requirements.txt', session='hack')
    return [str(ir.req) for ir in install_reqs if ir.name not in exclude]


def get_version_suffix():
    return os.environ.get('VERSION_SUFFIX')


def get_version(base_version):
    def version_parts():
        yield base_version
        suffix = get_version_suffix()
        if suffix:
            yield suffix

    return '.'.join(version_parts())


def remove_suffix(package):
    for v_sig in ['<=', '>=', '==']:
        pos = package.find(v_sig)
        if pos != -1:
            package = package[:pos]
    pos = package.find('[')
    if pos != -1:
        package = package[:pos]
    return package


def get_extra_indexes(require_file='requirements.txt'):
    url_pattern = re.compile(r'--extra-index-url=(\S+)')
    urls = []
    with open(require_file) as fin:
        for line in fin:
            match = url_pattern.match(line)
            if match:
                urls.append(match.group(1))
    return urls


def get_package_data_files(data_files):
    if data_files is None:
        data_files = []

    if not isinstance(data_files, (list, tuple)):
        raise ValueError(f"data_files must be a list or tuple.")

    return list(data_files)


def get_console_scripts(console_scripts):
    if console_scripts is None:
        console_scripts = {}

    if not isinstance(console_scripts, dict):
        raise ValueError(f"console_scripts must be a dict type, "
                         f"which the key is the script entry name, "
                         f"the value is the corresponding entry point.")

    return [f"{name} = {entry}" for name, entry in console_scripts.items()]


def perform_setup(package_name, version, data_files=None, console_scripts: dict = None):
    setup(
        name=package_name,
        version=get_version(version),
        author="Microsoft Crop",
        python_requires='>=3.6',
        platforms=('any', ),
        packages=find_packages(exclude=("*.tests.*", "*.test.*")),
        package_data={package_name: get_package_data_files(data_files)},
        include_package_data=True,
        install_requires=get_requirements(),
        dependency_links=get_extra_indexes(),
        entry_points={
            'console_scripts': get_console_scripts(console_scripts)
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: OS Independent",
        ],
    )

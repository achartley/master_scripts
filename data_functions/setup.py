from setuptools import setup

setup(
    name='master_data_functions',
    version='0.1.0',
    author='Geir Tore Ulvik',
    author_email='g.t.ulvik@fys.uio.no',
    description='Scripts and functions for scintillator data import and preparation',
    include_package_data=True,
    install_requires=[
        "Numpy >= 1.17",
    ],
)

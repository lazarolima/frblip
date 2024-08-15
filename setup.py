from setuptools import find_packages, setup

bla = setup(
    name='frblip',
    version='0.0.1',
    description='Fast Radio Burst mock catalogs synthesis.',
    author='Marcelo Vargas dos Santos',
    author_email='mvsantos_at_protonmail.com',
    packages=find_packages(include=['frblip', 'frblip.*']),
    include_package_data=True,
    package_data={'frblip': ['data/*.npy', 'data/*.npz', 'data/*.csv']},
    python_requires='>=3.10',
    install_requires=[
        'toolz',
        'dill==0.3.7',
        'numba==0.58.1',
        'numpy==1.26.0',
        'pandas==2.1.2',
        'sparse==0.13.0',
        'xarray==0.20.1',
        'scipy==1.11.2',
        'astropy',
        'astropy-healpix',
        'healpy',
        'pygedm',
        'camb',
        'pyccl==2.8.0',
    ],
)

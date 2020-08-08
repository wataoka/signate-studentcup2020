import os

HOME_DIR: str = os.path.expanduser('~')
HERE_DIR: str = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR: str = os.path.join(HOME_DIR, 'datasets', 'signate-studentcup2020')

BASE_DIR: str = HERE_DIR
UTILS_DIR: str = os.path.join(BASE_DIR, 'utils')
SUBMITS_DIR: str = os.path.join(BASE_DIR, 'submits')
import os


def create_dir(dir_name):
    """Creates directory if it does not exsit"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

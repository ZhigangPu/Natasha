import json
from utils.linux_wrapper import create_dir
from shutil import copyfile


class Config:

    """Class that loads hyperparameters from different source, like
    json file, dict object, list obejct.

    Attributes will be stored as class attribute.

    Take care that, currently, path must be ended with "/".

    Attributes:
        source: path of json file or list of path of json file or dict
                object.
    """

    def __init__(self, source):
        self.source = source

        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load_json(s)
        else:
            self.load_json(source)

    def load_json(self, source):
        """Read json k-v pairs into memory, namely, class's __dict__ data structure"""
        with open(source) as f:
            data = json.load(f)
            self.__dict__.update(data)

    def save(self, dir_name):
        """Save configurations to disk

        Configurations should contain 'export_name' attribute, and will be stored on the path
        'dirname + export_name'

        Args:
            dir_name: output directory
        """
        create_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            json.dumps(self.source, indent=4)
        else:
            copyfile(self.source, dir_name + self.export_name)

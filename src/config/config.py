import yaml
from yaml import YAMLError


class Config:
    def __init__(self, options: dict[str]):
        self.options = options

    @staticmethod
    def parse(filename: str):
        with open(filename, "r") as f:
            options = yaml.safe_load(f)
            return Config(options)


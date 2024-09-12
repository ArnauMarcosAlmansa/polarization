import argparse

from src.config.config import Config
from src.pipeline import Pipeline, sequential
import os

from src.tasks import train_nerfs


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", default="../../config.yaml")
    argparser.add_argument("--name", required=True)
    return argparser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config = Config.parse(args.config)
    # config = Config.parse("config.yaml")

    this_fle_path = os.path.dirname(os.path.realpath(__file__))

    train_nerfs_task = sequential(train_nerfs(config))

    pipeline = Pipeline([
        train_nerfs_task,
    ])

    pipeline.run()

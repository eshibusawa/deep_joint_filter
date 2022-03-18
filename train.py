import argparse

import add_path
from yaml_parser import YamlParser
from deep_joint_filter import DeepJointFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./configs/djf_rgbnir256_gaussion25.yaml", help="config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = YamlParser(config_file=args.config_file)

    djf  = DeepJointFilter(config)
    djf.load()
    djf.train()

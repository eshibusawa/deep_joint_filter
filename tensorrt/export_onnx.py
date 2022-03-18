import argparse

import torch
import torch.onnx

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

    batch_size = 1
    dummy_input=torch.randn(batch_size, 3, 255, 255).to('cuda')
    torch.onnx.export(djf.model.model, (dummy_input, dummy_input), "djf.onnx", verbose=False,
        input_names=["target_image", "guide_image"],
        output_names=["output_image"],
        dynamic_axes={"target_image" : {2 : 'image_height', 3 : 'image_width'},
                        "guide_image" : {2 : 'image_height', 3 : 'image_width'},
                            'output_image' : {2 : 'image_height', 3 : 'image_width'}})

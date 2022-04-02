import argparse
import os
import time

import numpy as np
import PIL.Image as Image
import torch

import add_path
from yaml_parser import YamlParser
from deep_joint_filter import DeepJointFilter

from inference_trt import TRTDataset
from inference_trt import TRTDatasetGT
from inference_trt import PSNR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/djf_rgbnir256_gaussion25.yaml', help='config file')
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--guide_dir', type=str)
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='/tmp')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlParser(config_file=args.config_file)
    if (args.target_dir is None) or (args.guide_dir is None):
        args.target_dir = config.dataset.test['target']
        args.guide_dir = config.dataset.test['guide']
        args.gt_dir = config.dataset.test['gt']

    os.makedirs(args.output_dir, exist_ok=True)

    # setup network
    djf  = DeepJointFilter(config)
    djf.load()
    input_dtype_np = np.float32

    # setup data loader
    gt_exists = args.gt_dir is not None
    if gt_exists:
        tds = TRTDatasetGT(args.target_dir, args.guide_dir, args.gt_dir, input_dtype_np)
    else:
        tds = TRTDataset(args.target_dir, args.guide_dir, input_dtype_np)

    psnrs = list()
    psnrNoises = list()
    inference_times = list()
    with torch.no_grad():
        for items, index in zip(tds, range(len(tds))):
            target = items[0]
            guide = items[1]

            t = time.time()
            t_target = torch.from_numpy(target).to(config.device)
            t_guide = torch.from_numpy(guide).to(config.device)
            t_output = djf.model.process(t_target, t_guide, None)
            output = t_output.to('cpu').numpy()
            inference_times.append(time.time() - t)
            if gt_exists:
                gt = tds.load_gt(index)
                psnrs.append(PSNR(gt, output))
                psnrNoises.append(PSNR(gt, target))

            output_uint8 = tds.postprocess(output.squeeze())
            img_output = Image.fromarray(output_uint8)
            fn_output = os.path.splitext(tds.load_name(index))[0] + '.png'
            img_output.save(os.path.join(args.output_dir, fn_output))

    average_inference_time = np.average(np.array(inference_times))
    if gt_exists:
        average_psnr = np.average(np.array(psnrs))
        average_psnrNoise = np.average(np.array(psnrNoises))
        print('Time: {}, PSNR {}, PSNRN {}'.format(average_inference_time, average_psnr, average_psnrNoise))
    else:
        print('Time: {}'.format(average_inference_time))

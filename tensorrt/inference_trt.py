import argparse
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import PIL.Image as Image

import add_path
from yaml_parser import YamlParser
from utils import load_flist

class TRTDataset():
    def __init__(self, target_dir, guide_dir, dtype):
        assert os.path.isdir(target_dir) and os.path.isdir(guide_dir)
        self.flist_target = load_flist(target_dir)
        self.flist_guide = load_flist(guide_dir)
        assert len(self.flist_target) == len(self.flist_guide)
        self.dtype = dtype
        self.index = 0

    @staticmethod
    def preprocess(img, dtype):
        array = None
        if img.mode == 'L':
            array = np.divide(np.array(img.convert('RGB')), 255, dtype=dtype, casting='safe')
        elif img.mode == 'RGB':
            array = np.divide(np.array(img), 255, dtype=dtype, casting='safe')
        return np.ascontiguousarray(array.transpose(2, 0, 1))[np.newaxis, :]

    @staticmethod
    def postprocess(array, to_byte=True):
        array = array.transpose(1, 2, 0)
        if to_byte:
            array = np.clip(array, 0, 1)
            array = (255 * array).astype(np.uint8)
        return array

    def __len__(self):
        return len(self.flist_target)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration()
        try:
            item = self.load_item(self.index)
        except:
            print('loading error: ' + self.load_name(self.index))
            item = self.load_item(0)
        self.index += 1
        return item

    def load_name(self, index):
        return os.path.basename(self.flist_target[index])

    def load_item(self, index):
        fn_target = self.flist_target[index]
        fn_guide = self.flist_guide[index]

        img_target = Image.open(fn_target)
        img_guide = Image.open(fn_guide)

        array_target = self.preprocess(img_target, self.dtype)
        array_guide = self.preprocess(img_guide, self.dtype)

        return  array_target, array_guide

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/djf_rgbnir256_gaussion25.yaml', help='config file')
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--guide_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='/tmp')
    parser.add_argument('--engine_file', type=str, default='./djf_engine.trt')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlParser(config_file=args.config_file)
    if (args.target_dir is None) or (args.guide_dir is None):
        args.target_dir = config.dataset.test['target']
        args.guide_dir = config.dataset.test['guide']

    # setup engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine_file, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # get shape and allocate memory
    maximum_shape = engine.get_profile_shape(0, 0)[2]
    input_dtype = engine.get_binding_dtype(0)
    if input_dtype == trt.float32:
        input_dtype_np = np.float32
    elif input_dtype == trt.float16:
        input_dtype_np = np.float16
    target = np.empty(maximum_shape, input_dtype_np)
    d_target = cuda.mem_alloc(target.nbytes)
    d_guide = cuda.mem_alloc(target.nbytes)
    d_output = cuda.mem_alloc(target.nbytes)

    # setup data loader
    tds = TRTDataset(args.target_dir, args.guide_dir, input_dtype_np)

    for items, index in zip(tds, range(len(tds))):
        target = items[0]
        guide = items[1]
        output = np.empty_like(target)
        bindings = [int(d_target), int(d_guide), int(d_output)]
        stream = cuda.Stream()

        context.set_binding_shape(0, target.shape)
        context.set_binding_shape(1, guide.shape)
        cuda.memcpy_htod_async(d_target, target, stream)
        cuda.memcpy_htod_async(d_guide, guide, stream)
        context.execute_async_v2(bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        output_uint8 = tds.postprocess(output.squeeze())
        img_output = Image.fromarray(output_uint8)
        fn_output = os.path.splitext(tds.load_name(index))[0] + '.png'
        img_output.save(os.path.join(args.output_dir, fn_output))

    d_target.free()
    d_guide.free()
    d_output.free()

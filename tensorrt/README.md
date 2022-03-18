# TensorRT [Japanese]

## root権限のinstall
### SDK Manager
```sh
dpkg -i /opt/jetson/sdkm_downloads/sdkmanager_1.7.1-8928_amd64.deb
apt-get install libgconf-2-4 libcanberra-gtk-module
dpkg -i /opt/jetson/sdkm_downloads/sdkmanager_1.7.1-8928_amd64.deb
```
Jeson Nano向けの設定を行うとJetPack 4.6とCUDA 10.2が選択される

### TensorRT
https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-debian
```sh
dpkg -i /opt/jetson/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt8.2.2.1-ga-20211214_1-1_amd64.deb
apt-get install tensorrt
apt-get install python3-libnvinfer-dev uff-converter-tf onnx-graphsurgeon
```

### Python仮想環境
```sh
apt-get install python3-virtualenv python3-pip
```

### CMakeの更新
```sh
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null |\
  gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
apt install cmake kitware-archive-keyring
rm /etc/apt/trusted.gpg.d/kitware.gpg
```

## user権限のinstall
Systemの`virtualenv`が古いためうまく動かない．
環境を2回作る．
### virtualenvを作成する
```sh
python -m virtualenv -p python3.6 ~/python_env/virtualenv-updated
source ~/python_env/virtualenv-updated/bin/activate
pip install -U virtualenv
python -m virtualenv --system-site-packages -p python3.6 ~/python_env/trt-20220214
deactivate
source ~/python_env/trt-20220214/bin/activate
```

最低限の確認
```sh
python -c 'import tensorrt; print("TensorRT version: {}".format(tensorrt.__version__))'
```

### PyCUDA
```sh
export CUDA_HOME=/usr/local/cuda
export PATH=${PATH}:${CUDA_HOME}/bin
pip install pycuda
```

import時に"system has unsupported display driver / cuda driver combination"なるerrorの場合は以下でGPU自体が動いているか確認．
```sh
nvidia-smi
```

### PyTorch
https://pytorch.org/get-started/locally/
```sh
pip3 install torch torchvision torchaudio
```
このmemoの時点(2022/2/14)ではCUDA 10.2向けのPyTorch 1.10.1が選択される
GPUが使えるか確認する
```sh
python -c 'import torch; print("PyTorch version: {}".format(torch.__version__)); print("GPU available: {}".format(torch.cuda.is_available()))'
```

### 動作確認 (PyTorch -> ONNX -> TensorRT engine -> TRT Python bindingsで実行)
適切な場所でsample codeをbuildする
```sh
mkdir tensorrt_sample
rsync -av /usr/src/tensorrt/. ./tensorrt_sample
make -C tensorrt_sample/sample -j8
```

基本的な流れは[Quick Start](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb)の通り


PythonからDeep Joint FilterのmodelをONNXへexportする
```sh
python export_onnx.py
```
`djf.onnx`が出力されたことを確認

TensorRTのengineを構築する
```sh
/path/to/tensorrt_sample/bin/trtexec --maxShapes=target_image:1x3x768x1024,guide_image:1x3x768x1024 --optShapes=target_image:1x3x682x1024,guide_image:1x3x682x1024 --minShapes=target_image:1x3x633x502,guide_image:1x3x633x502 --onnx=djf.onnx --saveEngine=djf_engine.trt --explicitBatch
/path/to/tensorrt_sample/bin/trtexec --maxShapes=target_image:1x3x768x1024,guide_image:1x3x768x1024 --optShapes=target_image:1x3x682x1024,guide_image:1x3x682x1024 --minShapes=target_image:1x3x633x502,guide_image:1x3x633x502 --onnx=djf.onnx --saveEngine=djf_engine_fp16.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```
`djf_engine.trt`と`djf_engine_fp16.trt`が出力されたことを確認
TensorRTで動作確認する
```sh
python interence_trt.py
```
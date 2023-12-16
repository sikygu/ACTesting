# ACTesting: Automated Cross-modal Testing Method of Text-to-Image Software

## Overview

This project aims to build a new Tool for testing T2I software generation results.  ACTesting is an Automated Cross-modal Testing Method of Text-to-Image software, the first testing method designed specifically for T2I software. We construct test samples based on entities and relationship triples following the fundamental principle of maintaining consistency in the semantic information to overcome the cross-modal matching challenges. To address the issue of testing oracle scarcity, we first design the metamorphic relation for T2I software and implement three types of mutation operators guided by adaptability density.

![](https://github.com/sikygu/ACTesting/blob/main/Overview.png)

## Environment Setup

We follow the project [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to setup the environment

### Requirements:

- Python <= 3.8
- PyTorch >= 1.2 (Mine 1.10.0 (CUDA 11.1))
- torchvision >= 0.4 (Mine 0.11.0 (CUDA 11.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV

### Step-by-step installation

```
conda create --name actesting
conda activate actesting
pip install ninja yacs cython matplotlib tqdm opencv-python overrides
```

```
# we use GPU 3090 with cuda 11.1, so we give the instructions for this. 
proxychains4 pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

```
mkdir ACTesting
# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../..
# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

```
# install Scene-Graph
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark
python setup.py build develop
```

## Data

- we use MS-COCO dataset to be the source seed. Download the coco2017val from [MS-COCO](http://mscoco.org/).

- The testing data we generated can be download from the [Baidu Cloud](https://pan.baidu.com/s/1e4MMVy_Nh5f6gYXYdOsevQ?pwd=fv94) with code fv94.

- We also provide the generated images from several software, which are also available from [Baidu Cloud](https://pan.baidu.com/s/1i6Bdvo0CCpXTPJOmHi1vaw?pwd=rdwx) with code rdwx. (The zip file is almost 7G)


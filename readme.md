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

```shell
conda create --name actesting
conda activate actesting
pip install ninja yacs cython matplotlib tqdm opencv-python overrides
```

```shell
# we use GPU 3090 with cuda 11.1, so we give the instructions for this. 
proxychains4 pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

```shell
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

```shell
# install Scene-Graph
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark
python setup.py build develop
```

## Data

- we use MS-COCO dataset to be the source seed. Download the coco2017val from [MS-COCO](http://mscoco.org/).

- The testing data we generated can be download from the [Baidu Cloud](https://pan.baidu.com/s/1e4MMVy_Nh5f6gYXYdOsevQ?pwd=fv94) with extraction code fv94.

- We also provide the generated images from several software, which are also available from [Baidu Cloud](https://pan.baidu.com/s/1i6Bdvo0CCpXTPJOmHi1vaw?pwd=rdwx) with extraction code rdwx. (The zip file is almost 7G). You can also use our testing data to generated images and put the results to /yourpath/ACTesting/images/yoursoftware/.

- The checkpoints for Scene-Graph can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/18vG1EHNLtPldcd3viS3xUA?pwd=vtg1) with extraction code vtg1.

- The generated images file can be organized as:

  /yourpath/ACTesting/images/yoursoftware/EC_captions_val2017/generatedimages.jpg

  /yourpath/ACTesting/images/yoursoftware/ER-R_captions_val2017/generatedimages.jpg

  where "yourpath" is your file path, "yoursoftware" is the T2I software name.

## Evaluation

#### I-FID:

```shell
python FID.py --path /yourpath/ACTesting/images/yoursoftware
```

#### I-IS:

```shell
python IS.py --image_folder /yourpath/ACTesting/images/yoursoftware
```

#### R-P:

use generateRPjson.ipynb to generate the RP files.

```shell
python RP.py --image_dir /yourpath/ACTesting/images/yoursoftware
```

#### Generate the scene graph:

Put run.sh to the child path /yourpath/ACTesting/Scene-Graph-Benchmark.pytorch and modify the paramaters.

```shell
cd Scene-Graph-Benchmark.pytorch
bash ./run.sh
cd ..
```

Then cacluate the error rate based on the MRs

```shell
python evaluation.py --model_path /yourpath/ACTesting/output/yoursoftware
```

## Results

![](https://github.com/sikygu/ACTesting/blob/main/rq2.png)

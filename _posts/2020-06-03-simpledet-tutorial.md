---
title: "Simpledet"
date: 2020-06-03 10:35:00 +0900
categories: simpledet object detection
---

For new simpledet researchers,   
```
https://github.com/TuSimple/simpledet
```

# Initial environment version   
OS: Linux-x86_64, Ubuntu 18.04   
NVIDIA Driver Version: 440.82   
CUDA 10.1   
cuDNN 7.6.5   

for checking the cuda environment,   
```
nvidia-smi
nvidia-settings
nvcc -V
```

# Installation with docker   
그래픽 드라이버, cuda, cuDNN ?   
docker 설치, nvidia-docker 설치  

```
nvidia-docker run -it -v $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR rogerchen/simpledet:cuda10 zsh   
```
이 구문, $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR
$HOST-SIMPLEDET-DIR
HOST에 올릴 작업 디렉토리, 저거 안치면 docker container에 기본 구조만 있음.

$CONTAINER-WORKDIR
docker container의 어디에다가 HOST-SIMPLEDET-DIR 파일들을 만들지 경로 지정할 수 있음

예를 들어서,
```
(base) dongjun@dongjun-System-Product-Name:~/djplace$ ls
beauvoir             labrado.jpg           synth-ml_0304.tar.xz  Untitled3.ipynb  Untitled.ipynb
blue_tit.jpeg        rename.py             synth-ml.tar.xz       Untitled4.ipynb  WRS_classifier
case3_test           simpledet             temps                 Untitled5.ipynb  wrs-data-collection
chessBoard.png       subtask_b.yaml        test                  Untitled6.ipynb  yamlMaker.py
German_Shepherd.jpg  synth-ml              Untitled1.ipynb       Untitled7.ipynb  yamlMaker.py~
kakaoArena           synth-ml_0303.tar.xz  Untitled2.ipynb       Untitled8.ipynb
(base) dongjun@dongjun-System-Product-Name:~/djplace$ nvidia-docker run -it -v "$(pwd)"/simpledet:/home/dongjun/djplace/simpledet rogerchen/simpledet:cuda10 zsh
root@7dd1f0cc95f8 /# cd home/dongjun/djplace/simpledet 
root@7dd1f0cc95f8 /h/d/d/simpledet# ls                                                                 master
LICENSE       README.md  detection_infer_speed.py  doc           operator_cxx  scripts   utils
MODEL_ZOO.md  config     detection_test.py         mask_test.py  operator_py   symbol
Makefile      core       detection_train.py        models        rpn_test.py   unittest

#nvidia-docker run -it -v "$(pwd)" rogerchen/simpledet:cuda10 zsh

nvidia-docker run -it -v "$(pwd)"/simpledet:"$(pwd)"/simpledet rogerchen/simpledet:cuda10 zsh

(with GPU setting)
```

In the docker container,   
```
# os version
lsb_release -a
'ubuntu 16.04'

# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

#or (checking)
#pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'

# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make

# mxnet version
python
# python 3.6.8
>>>import mxnet
>>>mxnet.__version__
'1.5.0'

# test simpledet installation
mkdir -p experiments/faster_r50v1_fpn_1x
python detection_infer_speed.py --config config/faster_r50v1_fpn_1x.py --shape 800 1333

#then you can see the number about the speed of detection_infer.
```

### Prepare the dataset   
```
# enter simpledet main directory
cd simpledet

# create data dir
mkdir -p data/src
pushd data/src

# download and extract clipart.zip
# courtesy to "Towards Universal Object Detection by Domain Attention"
wget https://1dv.aflat.top/clipart.zip -O clipart.zip
unzip clipart.zip
popd

# generate roidbs

data/label_map/voc_label_map.json # you have to make that voc_label_map.json, 
data/src/clipart/JPEGImages
data/src/clipart/Annotations
data/src/clipart/ImageSets

python3 utils/create_voc_roidb.py --data-dir data/src/clipart --split train
```

### Prepare to our own dataset
```
mkidr -p data/cache data/your_dataset

#set the paths for your own dataset
data/
    cache/
    your_dataset/
                annotations/
                            your_train.json
                            your_test.json
                images/
                      train/
                           *.png or *.jpg
                      test/
                           *.png or *.jpg
                      
                      
python utils/json_to_roidb.py --json data/your_dataset/your_xxxx.json
# and then your_xxxx.roidb will be created in data/cache/your_xxxx.roidb.roidb
# so you must change that name like this,
mv your_xxxx.roidb.roidb your_xxxx.roidb
```

If you met a problem at here,   
maybe you should check about the path of your_xxxx.json file.    

And, think that in your json file, there are path to your dataset images: "img_url".   
so you must set proper path about your images in your docker container.    

로컬 디렉토리에 json 파일이랑 이미지 데이터셋 만들어두고, docker 연동했다가 json 파일 안에 img_url이 로컬 디렉토리일 때의 이미지셋 경로를 가리키고 있는 것을 깜빡하고, roidb로 만들어서 detection_train.py 할 때 오류가 났었음.

if you met a problem at here,   
maybe you made something wrong typo to your path in json file or dataset.

And you need to change some line for you,      
```
# change the something_model_config to what you want to use
# like this, for 1 gpu setting
vi config/something_model_config.py
#gpus = [0, 1, 2, 3, 4, 5, 6, 7]
gpus = [0]

if train:
    #image_set = ("coco_train2017", )
    image_set = ("clipart_train", )
# config/__pycache__ 안에 있는 pyc 파일들을 지우고 다시 돌려준다.
cd ~/simpledet/config/__pycache__
rm -rf something_files.pyc
```

```
# run the detection_train file,
# --config config/faster_r50v1_fpn_1x.py
# It measn, I will use resnet-50-v1 pretrain model.
python detection_train.py --config config/faster_r50v1_fpn_1x.py

# Maybe here are some bugs about params,
cd pretrain
rm -rf param
#wget that_param_address, see MODEL_ZOO.md file
# download them yourself in, ~/simpledet/pretrain_model
wget https://1dv.aflat.top/resnet-v1-50-0000.params
wget https://1dv.aflat.top/resnet-v1-101-0000.params
wget https://1dv.aflat.top/resnet-50-0000.params
wget https://1dv.aflat.top/resnet-101-0000.params
wget https://1dv.aflat.top/resnet50_v1b-0000.params
wget https://1dv.aflat.top/resnet101_v1b-0000.params
wget https://1dv.aflat.top/resnet152_v1b-0000.params
wget https://1dv.aflat.top/resnext-101-64x4d-0000.params
wget https://1dv.aflat.top/resnext-101-32x8d-0000.params
wget https://1dv.aflat.top/resnext-152-32x8d-IN5k-0000.params
```

```
root@ffaba0b8053f /h/d/d/t/simpledet# python detection_train.py --config config/faster_r50v1_fpn_1x.py
[10:35:15] src/base.cc:84: Upgrade advisory: this mxnet has been built against cuDNN lib version 7500, which is older than the oldest version tested by CI (7600).  Set MXNET_CUDNN_LIB_CHECKING=0 to quiet this warning.
06-11 18:35:15 parameter shape
06-11 18:35:15 [('data', (2, 3, 800, 1333)),
 ('conv0_weight', (64, 3, 7, 7)),
 ('bn0_gamma', (64,)),
 ('bn0_beta', (64,)),
 ('bn0_moving_mean', (64,)),
...

06-11 19:35:28 Epoch[5] Batch [490]	Iter: 2990/3000	Lr: 0.00250	Speed: 1.71 samples/sec	Train-RpnAcc=0.997697,	RpnL1=0.239521,	RcnnAcc=0.955892,	RcnnL1=1.001961,	
06-11 19:35:39 Epoch[5] Train-RpnAcc=0.997648
06-11 19:35:39 Epoch[5] Train-RpnL1=0.239589
06-11 19:35:39 Epoch[5] Train-RcnnAcc=0.956123
06-11 19:35:39 Epoch[5] Train-RcnnL1=1.001152
06-11 19:35:39 Epoch[5] Time cost=601.038
06-11 19:35:39 Saved checkpoint to "experiments/faster_r50v1_fpn_1x/checkpoint-0006.params"
06-11 19:35:39 Training has done
06-11 19:35:49 Exiting
```

test할 때, 
```
cp -r simpledet/operator_py utils/
```


새로운 컨테이너를 불러오고, 계속 재설치를 하다보면 docker directory의 메모리가 반환되지 않을 때가 있다.   
심하다 싶으면 체크해준다.
```
docker system prune -a -f
``` 

---
title: "Simpledet"
date: 2020-06-03 10:35:00 +0900
categories: simpledet object detection
---

For simpledet users,   
```
https://github.com/TuSimple/simpledet
```

# Initial setting checking   
OS: Linux-x86_64, Ubuntu 18.04   
NVIDIA Driver Version: 440.82   
CUDA 10.1
cuDNN 7.6.5
쿠다 세팅 확인,   
```
nvidia-smi
nvidia-settings
nvcc -V
```

###
CUDA   | 10.0 |   10.1
cuDNN  | 7.6  |   7.6.5  
Result |      |     OK


# Installation with docker   
그래픽 드라이버, cuda, cuDNN ?   
docker 설치, nvidia-docker 설치  


# nvidia-docker run -it -v $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR rogerchen/simpledet:cuda10 zsh
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
# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

#or (확인중)
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"


pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'

# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make

# test simpledet installation
mkdir -p experiments/faster_r50v1_fpn_1x
python detection_infer_speed.py --config config/faster_r50v1_fpn_1x.py --shape 800 1333
```


GPU setting
```
# change the something_model_config to what you want to use
vi config/something_model_config.py
gpus = 0

if train:
    image_set = clipart_train
```


to check it out easily   
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

data/src/clipart/ 1 2 3 

python3 utils/create_voc_roidb.py --data-dir data/src/clipart --split train
```

```
fix pretrain/params using MODEL_ZOO.md
```




새로운 컨테이너를 불러오고, 계속 재설치를 하다보면 docker directory의 메모리가 반환되지 않을 때가 있다.   
심하다 싶으면 체크해준다.
```
docker system prune -a -f
``` 

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

###
CUDA   | 10.0 |
cuDNN  | 7.6  |   
Result |      |


# Installation with docker   
그래픽 드라이버, cuda 10.0, cuDNN 7.5 인가?   
docker 설치, nvidia-docker 설치  
```
docker login   


# nvidia-docker run -it -v $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR rogerchen/simpledet:cuda10 zsh
이 구문, $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR
$HOST-SIMPLEDET-DIR
HOST에 올릴 작업 디렉토리, 저거 안치면 docker container에 기본 구조만 있음.

$CONTAINER-WORKDIR
docker container의 어디에다가 HOST-SIMPLEDET-DIR 파일들을 만들지 경로 지정할 수 있음

예를 들어서,
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


쿠다 세팅 확인,   
```
nvidia-smi
nvidia-settings
nvcc -V
```










'''
pip install --upgrade pip 해주고,

pip install matplot 부분부터 시작   
```
pip install 'matplotlib<3.1' opencv-python pytz
pip install https://1dv.aflat.top/mxnet_cu100-1.6.0b20191214-py2.py3-none-manylinux1_x86_64.whl

# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

#or (확인중)
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"


# install mxnext, a wrapper around MXNet symbolic API
pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'
'''


# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make

# test simpledet installation
mkdir -p experiments/faster_r50v1_fpn_1x
python detection_infer_speed.py --config config/faster_r50v1_fpn_1x.py --shape 800 1333
```

새로운 컨테이너를 불러오고, 계속 재설치를 하다보면 docker directory의 메모리가 반환되지 않을 때가 있다.   
심하다 싶으면 체크해준다.
```
docker system prune -a -f
``` 

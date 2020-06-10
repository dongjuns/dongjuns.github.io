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
docker, nvidia-docker   
docker login   
nvidia-docker run --gpus all -it rogerchen/simpledet:cuda10 zsh   
(with GPU setting)

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

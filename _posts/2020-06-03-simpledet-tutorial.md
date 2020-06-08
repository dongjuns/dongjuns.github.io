---
title: "Simpledet"
date: 2020-06-03 10:35:00 +0900
categories: simpledet object detection
---

For simpledet users,   
```
https://github.com/TuSimple/simpledet
```

# Installation with docker   
그래픽 드라이버, cuda 10.0, cuDNN 7.5 인가?   
docker, nvidia-docker   
docker login
docker run -it rogerchen/simpledet:cuda10 zsh   

쿠다 세팅 확인,   
```
nvidia-smi
nvidia-settings
nvcc -V
```

pip install --upgrade pip 해주고,

pip install matplot 부분부터 시작   
```
pip install 'matplotlib<3.1' opencv-python pytz
pip install https://1dv.aflat.top/mxnet_cu100-1.6.0b20191214-py2.py3-none-manylinux1_x86_64.whl --user

# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

# install mxnext, a wrapper around MXNet symbolic API
pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'

# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make

# test simpledet installation
mkdir -p experiments/faster_r50v1_fpn_1x
python detection_infer_speed.py --config config/faster_r50v1_fpn_1x.py --shape 800 1333
```


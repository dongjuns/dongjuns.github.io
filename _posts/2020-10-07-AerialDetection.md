---
title: "Recipe for Aerial Detection"
date: 2020-10-07 12:56:00 +0900
categories: AerialDetection Dacon 
---

<https://dacon.io/competitions/official/235644/overview/>

Let's slice it to four parts.

# Install

cuda 10.0
torch 1.1

```
conda create -n arirang python=3.7 -y
source activate arirang

pip install cython
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

git clone https://github.com/dingjiansw101/AerialDetection.git
cd AerialDetection

sudo chmod 777 compile.sh
./compile.sh
pip install -r requirements.txt
pip install mmcv-full
python setup.py develop

sudo apt-get install swig
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```


# Train



# Test



# Sumbission

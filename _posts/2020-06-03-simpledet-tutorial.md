---
title: "Simpledet simple guide"
date: 2020-06-03 10:35:00 +0900
categories: simpledet object_detection
---

For new simpledet researchers with docker,   
<https://github.com/TuSimple/simpledet>   

---
# How to install SimpleDet?
## Initial environment versions
Before the installation SimpleDet,    
we need to set up environments.    
Here are my environment versions.
- Linux-x86_64, Ubuntu 18.04   
- NVIDIA Driver Version: 440.82   
- CUDA 10.1   
- cuDNN 7.6.5   

We will install initial environment in order, like as Graphics driver > CUDA > cuDNN > Docker.    
Then we can use SimpleDet by docker image.    

### Graphic settings installation
Refer to this link for installation GPU environment such as NVIDIA graphic driver, CUDA and cuDNN.    
<https://dongjuns.github.io/machine/learning/Machine-Learning-Set-up/>


If you want to check information about your GPU,    
there are some commands.    
You can trust the number of Driver Version,    
but don't easily trust the number of CUDA version by 'nvidia-smi'.    
Use these commands just to be sure,
- 'nvidia-settings' for graphic card driver and more graphic information by GUI style    
- 'nvcc -V' for CUDA version check    

```
#nvidia-settings
#nvcc -V

(base) dongjun@dongjun-System-Product-Name:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Fri_Feb__8_19:08:17_PST_2019
Cuda compilation tools, release 10.1, V10.1.105
```

- 'watch nvidia-smi' for monitoring GPU status while you are using it.    
```
#nvidia-smi
#watch nvidia-smi
(base) dongjun@dongjun-System-Product-Name:~$ watch nvidia-smi
Fri Jun 19 09:54:26 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 00000000:01:00.0 Off |                  N/A |
| 22%   43C    P5    20W / 250W |    932MiB / 12212MiB |      7%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1294      G   /usr/lib/xorg/Xorg                           395MiB |
|    0      1560      G   /usr/bin/gnome-shell                         266MiB |
|    0      2378      G   ...AAAAAAAAAAAAAAgAAAAAAAAA --shared-files   261MiB |
+-----------------------------------------------------------------------------+
```

### Docker installation
I and SimpleDet guys would strongly recommend you to use SimpleDet by docker image.    
Don't worry even if you are not docker person.    
Refer to this link,    
<https://dongjuns.github.io/machine/learning/Machine-Learning-Set-up/>

---

## SimpleDet with docker
Finally we arrived here,    
then we can use SimpleDet docker image!    
<https://github.com/TuSimple/simpledet/blob/master/doc/INSTALL.md>
```
nvidia-docker run -it -v $HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR rogerchen/simpledet:cuda10 zsh   
```

You can use the docker image instantly, like this.    
```
nvidia-docker run -it rogerchen/simpledet:cuda10 zsh
```

But with that option '-v', we can share the files between the docker container and your own workspace.    
Docker container has volatility, so we can't save our work result after working from docker container, basically.     
But with '-v' option, we can connect between the docker container and your own workspace.    

And look at this line,    
'$HOST-SIMPLEDET-DIR:$CONTAINER-WORKDIR'    
- $HOST-SIMPLEDET-DIR: Docker container에서도 사용하고 싶은 디렉토리, 안입력하면 default docker container 이용.    
- $CONTAINER-WORKDIR: Docker container에 경로를 지정해서 HOST-SIMPLEDET-DIR의 파일들을 사용할 수 있음.    

```
(base) dongjun@dongjun-System-Product-Name:~/djplace$ ls
simpledet             temps               wrs-data-collection

(base) dongjun@dongjun-System-Product-Name:~/djplace$ nvidia-docker run -it -v "$(pwd)"/simpledet:/home/dongjun/djplace/simpledet rogerchen/simpledet:cuda10 zsh
root@7dd1f0cc95f8 /# cd home/dongjun/djplace/simpledet 
root@7dd1f0cc95f8 /h/d/d/simpledet# ls                                                                 master
LICENSE       README.md  detection_infer_speed.py  doc           operator_cxx  scripts   utils
MODEL_ZOO.md  config     detection_test.py         mask_test.py  operator_py   symbol
Makefile      core       detection_train.py        models        rpn_test.py   unittest

#nvidia-docker run -it -v "$(pwd)" rogerchen/simpledet:cuda10 zsh
#nvidia-docker run -it -v "$(pwd)"/simpledet:"$(pwd)"/simpledet rogerchen/simpledet:cuda10 zsh
```

After connected to the docker container,
```
# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

#or (checking)
#pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'

# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make
```

You can also check the versions in the docker container.
```
# OS version
lsb_release -a
'ubuntu 16.04'

#Python version
python --version

# MXnet version
python
>>>import mxnet
>>>mxnet.__version__
'1.5.0'

# check out the installation status of simpledet
mkdir -p experiments/faster_r50v1_fpn_1x
python detection_infer_speed.py --config config/faster_r50v1_fpn_1x.py --shape 800 1333

#then you can see the number about the speed of detection_infer
```

새로운 컨테이너를 불러오고, 계속 재설치를 하다보면 docker directory의 메모리가 반환되지 않을 때가 있다.   
심하다 싶으면 체크해준다.    

Sometimes, if there is a stuck, docker can't return the memory for you.    
So when you think of too much memory used without reasonable reason,    
once remove the docker volumes.
```
docker system prune -a -f
``` 


# How to get dataset?
Before doing something by your deep learning model, you first need to prepare dataset.    
Refer to Simpledet, there is simple and easy guidline to get dataset, but you must meet some problems on there.    

### clipart dataset
I think COCO dataset is so bigger to practice,    
so I recommend to use the clipart dataset, firstly.
```
# enter simpledet main directory
cd simpledet

# create data dir
mkdir -p data/src
pushd data/src

# download and extract clipart.zip
# courtesy to "Towards Universal Object Detection by Domain Attention"
wget -c https://1dv.aflat.top/clipart.zip -O data/src/clipart.zip
unzip clipart.zip -d data/clipart
popd

# generate roidbs
data/label_map/voc_label_map.json # you have to make that voc_label_map.json, 
data/src/clipart/JPEGImages
data/src/clipart/Annotations
data/src/clipart/ImageSets

python3 utils/create_voc_roidb.py --data-dir data/src/clipart --split train
```
And you must meet an error like this below, at the last line.
```
the following error：
Traceback (most recent call last):
File "utils/create_voc_roidb.py", line 80, in
create_roidb(*parse_args())
File "utils/create_voc_roidb.py", line 19, in parse_args
with open(args.label_map) as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data/label_map/voc_label_map.json'
```
So you have to generate voc_label_map.json,    
use this script in data/src/clipart
```
import os
import json
import cv2

path = os.getcwd()
AnnotationsPath = os.listdir(os.path.join(path, "Annotations"))

from xml.etree.ElementTree import parse

words = []
for file in AnnotationsPath:
    tree = parse(os.path.join(path, "Annotations", file))
    root = tree.getroot() 
    objects = root.findall("object")
    names = [x.findtext("name") for x in objects]
    
    for name in names:
        if name not in words:
            words.append(name)
    

tempDictionary = {}
train_id = 1
for word in words:
    tempDictionary[word] = train_id
    train_id += 1
    
print(tempDictionary)
with open("voc_label_map.json", "w") as write_file:
    json.dump(tempDictionary, write_file)
```
And then, place it into data/label_map/voc_label_map.json


### our own dataset
```
mkidr -p data/cache data/your_dataset

# set the paths for your own dataset
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
                      
                      
python utils/json_to_roidb.py --json data/your_dataset/your_dataset.json
# and then your_dataset.roidb will be created in data/cache/your_dataset.roidb.roidb

# so you must change that name like this below,
mv your_dataset.roidb.roidb your_dataset.roidb
```

If you met some problems at here,   
maybe you should check about the path of your_dataset.json file.    

And, look at your json file,    
there are path about images of your dataset: "img_url".   
You must set the proper path about your images in your docker container.    

로컬 디렉토리에 json 파일이랑 이미지 데이터셋 만들어두고, docker 연동했다가 json 파일 안에 img_url이 로컬 디렉토리일 때의 이미지셋 경로를 가리키고 있는 것을 깜빡하고, roidb로 만들어서 detection_train.py 할 때 오류가 났었음.

In my case, I first made my json file so there were my local path about img_url in json file,    
so in docker container utils/json_to_roidb.py coudln't work.


FOR ME: wrs_json.py 사용할때, count 0 이랑 200 바꿔주는 거 주의.

if you met a problem at here,   
maybe you made something wrong typo to your path in json file or dataset.


### Modify some lines in config/something_model_config.py
They recommend you to copy the config file and modify it to use.    
And we need change some lines for using it well.
```
# change the something_model_config to what you want to use
# for your proper gpu setting 
# I used one gpu
vi config/something_model_config.py
#gpus = [0, 1, 2, 3, 4, 5, 6, 7]
gpus = [0]

...

# change the num_reg_class for regression
# num_reg_class = 81 # 1 + 80, it means there are 80 classes in the dataset and 1 for background.
# but some models don't work like this. so just leave them, like as 2 or something else.

num_reg_class = number of your classes + 1

#I have 12 classes, so
num_reg_class = 1 + 12

...

# change the num_class for classfication
#num_class = 80 + 1
num_class = number of your classes + 1

# in my case, it also
num_class = 12 + 1
...


# in my case, my roidb name is wrs_train.roidb and wrs_test.roidb in data/cache/
class DatasetParam:
    if is_train:
        #image_set = ("coco_train2017", )
        #image_set = ("your_own_dataset_train_roidb_name", )
        image_set = ("wrs_train", ) #in my case
    else:
        #image_set = ("coco_val2017", )
        #image_set = ("your_own_dataset_test_roidb_name",  )
        image_set = ("wrs_test", ) #in my case

...
```

After modifying the config files, sometime we need to remove .pyc files. 
```
# config/__pycache__ 안에 있는 pyc 파일들을 지우고 다시 돌려준다.
cd ~/simpledet/config/__pycache__
rm -rf something_config_files.pyc
```


Explain the options while runinng the detection_train file
```
# run the detection_train file,
# --config config/faster_r50v1_fpn_1x.py
# That name means, I will use fatset RCNN with resnet-50-v1 pretrain model.
python detection_train.py --config config/faster_r50v1_fpn_1x.py

# Maybe there are some bugs by broken params,
cd pretrain
rm -rf params

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

### GPU setting error
We need to set the proper number of GPU.
```
root@c8f7c6964f15 /h/d/d/d/simpledet# python detection_train.py --config config/faster_r50v
1_fpn_1x.py

... # doing well but error occured on here

06-15 15:48:00 warmup lr 0.006666666666666667, warmup step 500
Traceback (most recent call last):
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/symbol/symbol.py", line 1623, in simple_bind
    ctypes.byref(exe_handle)))
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet
/base.py", line 253, in check_call
    raise MXNetError(py_str(_LIB.MXGetLastError()))
mxnet.base.MXNetError: [07:48:03] src/engine/./../common/cuda_utils.h:318: Check failed: e 
== cudaSuccess || e == cudaErrorCudartUnloading: CUDA: invalid device ordinal
Stack trace:
  [bt] (0) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(dmlc::LogMessageFatal::~LogMessageFatal()+0x32) [0x7f725b4b99c2]
  [bt] (1) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(mxnet::common::cuda::DeviceStore::SetDevice(int)+0xd8) [0x7f725d6f0e98]
  [bt] (2) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(mxnet::common::cuda::DeviceStore::DeviceStore(int, bool)+0x48) [0x7f725d6f0f08]
  [bt] (3) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(mxnet::storage::GPUPooledStorageManager::Alloc(mxnet::Storage::Handle*)+0x10e) [0x7f725d71492e]
  [bt] (4) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(mxnet::StorageImpl::Alloc(mxnet::Storage::Handle*)+0x57) [0x7f725d717257]
  [bt] (5) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(+0x2be26f9) [0x7f725cfa26f9]
  [bt] (6) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mx
net/libmxnet.so(mxnet::NDArray::Chunk::Chunk(mxnet::TShape, mxnet::Context, bool, int)+0x19
8) [0x7f725cfc7598]
  [bt] (7) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::NDArray::NDArray(mxnet::TShape const&, mxnet::Context, bool, int)+0x
97) [0x7f725cfc7887]
  [bt] (8) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::common::InitZeros(mxnet::NDArrayStorageType, mxnet::TShape const&, mxnet::Context const&, int)+0x58) [0x7f725cfc7aa8]
During handling of the above exception, another exception occurred:



Traceback (most recent call last):
  File "detection_train.py", line 311, in <module>
    train_net(parse_args())
  File "detection_train.py", line 293, in train_net
    profile=profile
  File "/home/dongjun/djplace/docker_work/simpledet/core/detection_module.py", line 969, in fit
    for_training=True, force_rebind=force_rebind)
  File "/home/dongjun/djplace/docker_work/simpledet/core/detection_module.py", line 450, in bind
    state_names=self._state_names)
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/module/executor_group.py", line 280, in __init__
    self.bind_exec(data_shapes, label_shapes, shared_group)
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/module/executor_group.py", line 376, in bind_exec
    shared_group))
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/module/executor_group.py", line 670, in _bind_ith_exec
    shared_buffer=shared_data_arrays, **input_shapes)
  File "/root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/symbol/symbol.py", line 1629, in simple_bind
    raise RuntimeError(error_msg)
RuntimeError: simple_bind error. Arguments:
data: (2, 3, 800, 1333)
gt_bbox: (2, 100, 5)
im_info: (2, 3)
rpn_cls_label: (2, 267069)
rpn_reg_target: (2, 12, 89023)
rpn_reg_weight: (2, 12, 89023)
[07:48:03] src/engine/./../common/cuda_utils.h:318: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading: CUDA: invalid device ordinal
Stack trace:
  [bt] (0) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(dmlc::LogMessageFatal::~LogMessageFatal()+0x32) [0x7f725b4b99c2]
  [bt] (1) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::common::cuda::DeviceStore::SetDevice(int)+0xd8) [0x7f725d6f0e98]
  [bt] (2) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::common::cuda::DeviceStore::DeviceStore(int, bool)+0x48) [0x7f725d6f0f08]
  [bt] (3) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::storage::GPUPooledStorageManager::Alloc(mxnet::Storage::Handle*)+0x10e) [0x7f725d71492e]
  [bt] (4) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::StorageImpl::Alloc(mxnet::Storage::Handle*)+0x57) [0x7f725d717257]
  [bt] (5) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(+0x2be26f9) [0x7f725cfa26f9]
  [bt] (6) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::NDArray::Chunk::Chunk(mxnet::TShape, mxnet::Context, bool, int)+0x198) [0x7f725cfc7598]
  [bt] (7) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::NDArray::NDArray(mxnet::TShape const&, mxnet::Context, bool, int)+0x97) [0x7f725cfc7887]
  [bt] (8) /root/.pyenv/versions/3.6.8/lib/python3.6/site-packages/mxnet-1.5.0-py3.6.egg/mxnet/libmxnet.so(mxnet::common::InitZeros(mxnet::NDArrayStorageType, mxnet::TShape const&, mxnet::Context const&, int)+0x58) [0x7f725cfc7aa8]


terminate called without an active exception
[1]    29004 abort (core dumped)  python detection_train.py --config config/faster_r50v1_fpn_1x.py

# Change the number of GPU for your proper setting
#gpus = [0, 1, 2, 3, 4, 5, 6, 7]
gpus = [0]
```


### Training the object detection model
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
in the config/file.py, you can change other options for data processing.    
epoch, resizeParam, mean, std... but you know, need to figure out that well.    
Each model has its own architecture.


### Testing the object detection model
test할 때, operator_py Error가 뜨면 simpledet/operator_py 디렉토리를 simpledet/utlils 디렉토리 안에 복사해준다.
If there is operator_py error, cp -r simpledet/operator_py simpledet/utils/
```
cp -r operator_py utils/
```

Modify some lines like as one below,
```
vi detection_test.py

...

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from utils.roidb_to_coco import roidb_to_coco
    #if pTest.coco.annotation is not None:
    #    coco = COCO(pTest.coco.annotation)
    #else:
    coco = roidb_to_coco(roidbs_all)
```

### Testing the object detection model, really
```
root@c8f7c6964f15 /h/d/d/d/simpledet# python detection_test.py --config config/retina_r50v1
_fpn_1x.py
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
evaluating [0, 1000)
total number of images: 50
load experiments/retina_r50v1_fpn_1x/checkpoint-0006.params
parameter shape
[('rec_id', (1,)),
 ('im_id', (1,)),
 ('im_info', (1, 3)),
 ('data', (1, 3, 800, 1280)),
 ('conv0_weight', (64, 3, 7, 7)),
 ('bn0_gamma', (64,)),
...
convert to coco format uses: 0.1
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.46s).
Accumulating evaluation results...
DONE (t=0.10s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.705
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.873
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.805
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.739
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.802
coco eval uses: 0.7
```

'coco eval uses' shows that the number of time cost for evaluation.

---


My result    
All AP are resulted on wrs-data-collection,    
- training set: 200
- test set: 50


|Model|Backbone|Head|Train Schedule|1|2|3|4|5|6|7|8|9|10|11|12|mAP|
|-----|--------|----|--------------|-|-|-|-|-|-|-|-|-|--|--|--|---|
|Faster R-CNN |R50v1b-FPN|2MLP|1X|0.768|0.822|0.782|0.899|0.886|0.830|0.708|0.555|0.860|0.804|0.745|0.698|0.780|
|Faster R-CNN|R50v1b-FPN|2MLP|2X|0.794|0.900|0.779|0.919|0.877|0.899|0.732|0.664|0.855|0.792|0.776|0.783|0.814|
|Faster R-CNN|R101v1b-FPN|2MLP|1X|0.764|0.897|0.743|0.877|0.866|0.906|0.702|0.635|0.844|0.716|0.729|0.755|0.786|
|Faster R-CNN|R101v1b-FPN|2MLP|2X|0.791|0.865|0.792|0.909|0.890|0.909|0.729|0.671|0.867|0.787|0.762|0.772|0.812|
|RetinaNet|R50v1b-FPN|4Conv|1X|0.785|0.915|0.759|0.888|0.905|0.906|0.420|0.408|0.828|0.780|0.720|0.670|0.749|
|RetinaNet|R101v1b-FPN|4Conv|1X|0.789|0.888|0.778|0.913|0.877|0.859|0.514|0.523|0.801|0.723|0.647|0.721|0.753|
|TridentNet|R50v1b-C4|C5-128ROI|1X|0.562|0.697|0.272|0.571|0.299|0.536|0.007|0.000|0.327|0.236|0.023|0|0.294|
|TridentNet|R50v1b-C4|C5-128ROI|2X|0.647|0.694|0.552|0.733|0.698|0.708|0.279|0.170|0.755|0.659|0.630|0.634|0.597|
|TridentNet|R101v1b-C4|C5-128ROI|1X|0.531|0.496|0.318|0.592|0.397|0.521|0.034|0.000|0.240|0.304|0.306|0.244|0.332|
|TridentNet|R101v1b-C4|C5-128ROI|2X|0.564|0.738|0.636|0.648|0.766|0.736|0.321|0.234|0.719|0.653|0.626|0.628|0.606|

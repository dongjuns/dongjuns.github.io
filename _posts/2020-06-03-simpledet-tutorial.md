---
title: "Simpledet simple guide"
date: 2020-06-03 10:35:00 +0900
categories: simpledet object detection
---

For new simpledet researchers with docker,   
<https://github.com/TuSimple/simpledet>   


# Install SimpleDet
## Initial environment versions
Before the installation SimpleDet, we need to set up environments.    
Here are my environment versions, just to be sure.    

- OS: Linux-x86_64, Ubuntu 18.04   
- NVIDIA Driver Version: 440.82   
- CUDA 10.1   
- cuDNN 7.6.5   


### Installation for graphic settings
그래픽 드라이버    
cuda    
cuDNN    
```
link to install them
```

If you want to check information related to your GPU, there are some commands.    
you can trust the number of Driver Version, But don't trust the number of CUDA version with command 'nvidia-smi'.    
Use these    
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

And use 'watch nvidia-smi' for monitoring GPU status while you are using it.    
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


### Installation for Docker   
I and they, SimpleDet guys, strongly recomment to install SimpleDet with docker.    
Don't worry even if you are not docker person.    

how to install docker 설치    
nvidia-docker 설치  


## Installation SimpleDet with docker   
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

```
# voc_label_map.json codes 넣기
```


FOR ME: wrs_json.py 사용할때, count 0 이랑 200 바꿔주는 거 주의.

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

# change the number of classes
#num_reg_class = 81 # it means there are 80 classes in the dataset.
num_reg_clas = number of your classes + 1

...

#num_class = 80 + 1
num_class = number of your classes + 1

...

if train:
    #image_set = ("coco_train2017", )
    #image_set = ("your_own_dataset_roidb_name", )
    # in my case, my roidb name is wrs_train.roidb and wrs_test.roidb in data/cache/
    image_set = ("wrs_train",)
    
    
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

### GPU setting error
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

# Change the GPU setting like above,
```


### training the object detection model
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
resizeParam, mean, std...    



### test the object detection model
test할 때, operator_py Error가 뜨면 simpledet/operator_py 디렉토리를 simpledet/utlils 디렉토리 안에 복사해준다.
```
cp -r operator_py utils/
```

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

새로운 컨테이너를 불러오고, 계속 재설치를 하다보면 docker directory의 메모리가 반환되지 않을 때가 있다.   
심하다 싶으면 체크해준다.
```
docker system prune -a -f
``` 


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

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

|Model|Backbone|Head|Train Schedule|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Faster|R50v1b-C4|C5-512ROI|1X|78.0|98.6|93.0|-100.0|69.6|78.5|
|Faster|R50v1b-C4|C5-512ROI|2X|||||||
|Faster|R101v1b-C4|C5-512ROI|1X|||||||
|Faster|R101v1b-C4|C5-512ROI|2X|||||||
|Faster|R152v1b-C4|C5-512ROI|1X|||||||
|Faster|R152v1b-C4|C5-512ROI|2X|||||||
|Faster|R50v1b-FPN|2MLP|1X|78.0|98.6|93.0|-100.0|69.6|78.5|
|Faster|R50v1b-FPN|2MLP|2X|38.0|59.7|41.5|22.2|41.6|48.8|
|Faster|R101v1b-FPN|2MLP|1X|39.9|62.1|43.5|23.1|44.4|51.1|
|Faster|R101v1b-FPN|2MLP|2X|40.4|62.1|44.0|23.2|44.4|52.7|
|Faster|R152v1b-FPN|2MLP|1X|41.5|63.5|45.7|24.7|46.0|53.3|
|Faster|R152v1b-FPN|2MLP|2X|42.0|63.6|45.9|24.8|45.9|55.0|
|Mask(BBox)|R50v1b-FPN|2MLP|1X|37.8|59.9|40.9|22.9|41.5|48.0|
|Mask(BBox)|R50v1b-FPN|2MLP|2X|38.6|60.3|41.8|22.6|42.4|49.8|
|Mask(BBox)|R101v1b-FPN|2MLP|1X|40.4|62.2|44.1|24.0|44.4|52.1|
|Mask(BBox)|R101v1b-FPN|2MLP|2X|41.3|62.8|45.0|23.9|45.4|53.7|
|Mask(BBox)|R152v1b-FPN|2MLP|1X|41.8|63.7|46.1|25.3|46.3|53.6|
|Mask(BBox)|R152v1b-FPN|2MLP|2X|42.8|63.8|46.8|24.6|47.1|55.9|
|Mask(Inst)|R50v1b-FPN|2MLP|1X|34.4|56.5|36.2|18.7|37.9|46.4|
|Mask(Inst)|R50v1b-FPN|2MLP|2X|34.9|56.9|37.1|18.3|38.4|47.8|
|Mask(Inst)|R101v1b-FPN|2MLP|1X|36.3|58.8|38.6|19.4|39.7|49.7|
|Mask(Inst)|R101v1b-FPN|2MLP|2X|36.9|59.3|39.4|19.1|40.7|51.0|
|Mask(Inst)|R152v1b-FPN|2MLP|1X|37.4|60.1|39.8|20.0|41.6|50.7|
|Mask(Inst)|R152v1b-FPN|2MLP|2X|38.0|60.6|40.6|19.8|41.9|52.8|
|Trident|R50v1b-C4|C5-128ROI|1X|38.4|59.7|41.5|21.4|43.6|52.4|
|Trident|R50v1b-C4|C5-128ROI|2X|39.6|60.9|42.9|22.5|44.5|53.9|
|Trident|R101v1b-C4|C5-128ROI|1X|42.2|63.6|45.3|24.5|47.2|57.7|
|Trident|R101v1b-C4|C5-128ROI|2X|43.0|64.3|46.3|25.3|47.9|58.4|
|Trident|R152v1b-C4|C5-128ROI|1X|43.7|64.1|48.0|26.9|47.9|58.9|
|Trident|R152v1b-C4|C5-128ROI|2X|44.4|65.4|48.3|26.4|49.4|59.6|
|TridentFast|R50v1b-C4|C5-128ROI|1X|37.7|58.7|40.3|19.5|42.4|52.7|
|TridentFast|R50v1b-C4|C5-128ROI|2X|39.0|60.2|41.8|20.8|43.6|53.8|
|TridentFast|R101v1b-C4|C5-128ROI|1X|41.1|62.5|43.9|22.1|45.7|57.7|
|TridentFast|R101v1b-C4|C5-128ROI|2X|42.5|63.7|46.0|23.3|46.7|59.3|
|TridentFast|R152v1b-C4|C5-128ROI|1X|42.7|64.0|45.6|23.4|47.5|59.1|
|TridentFast|R152v1b-C4|C5-128ROI|2X|43.9|65.1|47.0|25.1|48.1|60.4|
|Retina|R50v1b-FPN|4Conv|1X|36.6|56.9|39.0|20.3|40.7|47.2|
|Retina|R101v1b-FPN|4Conv|1X|39.2|59.5|42.2|22.8|44.0|51.1|
|Retina|R152v1b-FPN|4Conv|1X|40.4|61.1|43.4|23.6|45.0|52.3|
|Faster|R50v1b-C4-DCNv1|C5-512ROI|1X|38.8|60.0|41.3|20.6|43.3|53.2|
|Faster|R101v1b-C4-DCNv1|C5-512ROI|1X|41.4|63.0|44.7|22.7|46.1|56.8|
|Faster|R50v1b-C4-DCNv2|C5-512ROI|1X|39.6|60.8|42.7|20.8|43.9|54.2|
|Faster|R50v1b-C4-DCNv2|C5-512ROI|2X|41.2|62.2|44.7|21.7|45.3|57.0|
|Faster|R101v1b-C4-DCNv2|C5-512ROI|1X|41.7|63.0|44.7|22.8|46.1|57.3|
|Faster|R101v1b-C4-DCNv2|C5-512ROI|2X|42.7|63.7|46.0|24.9|46.9|57.9|
|Retina|R50v1b-FPN-TR152v1b1X|4Conv|1X|38.9|59.0|41.6|21.4|43.3|52.1|
|Retina|R50v1b-FPN-TR152v1b1X|4Conv|2X|40.1|60.6|43.1|21.8|44.5|54.3|
|Faster|R50v1b-FPN-TR152v1b2X|2MLP|1X|39.9|61.3|43.6|22.7|44.2|52.7|
|Faster|R50v1b-FPN-TR152v1b2X|2MLP|2X|40.5|62.2|43.9|23.1|44.7|53.9|

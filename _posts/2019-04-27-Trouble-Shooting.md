---
title: "Trouble Shooting"
date: 2019-04-27 21:19:00 +0900
categories: Trouble Shooting
---

### import matplotlib
주피터 노트북을 사용할 때, matplotlib을 import하다가 에러날 때가 있다.
```
In [] import matplotlib.pyplot as plt
Users/jeongdongjun/anaconda3/lib/python3.7/site-packages/matplotlib/font_manager.py:232: UserWarning: 
Matplotlib is building the font cache using fc-list. This may take a moment.   
'Matplotlib is building the font cache using fc-list. '
```

Solution
```
$cd ~/.matplotlib
$rm -rf tex.cache
```
- - -
### jupyter에서 tensorflow import 에러
주피터 노트북을 사용할 때, tensorflow를 import하다가 에러날 때가 있다.
```
In []import tensorflow as tf

Out [] ---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-1-88d96843a926> in <module>
----> 1 import tensorflow as tf

ModuleNotFoundError: No module named 'tensorflow'

In []conda install tensorflow
#and restart the jupyter notebook
```

Solution
```
In []import tensorflow as tf
In []tf.__version__
Out []'1.13.1'
```
- - -
### Tensorflow 
Linear regression의 descent 관련,
```
...
update = W.assign(descent)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-18-3aca239bc8a4> in <module>
----> 1 update = W.assign(descent)# Launch the graph in a session.
      2 with tf.Session() as sess:
      3     # Initializes global variables in the graph.
      4     sess.run(tf.global_variables_initializer())
      5 

AttributeError: 'Tensor' object has no attribute 'assign'
...
```
W를 Tensor로 지정해서 그렇다. variable로 선언해줘야한다.
```
W = tf.placeholder(tf.float32)
#변경
W = tf.Variable(tf.random_normal([1]), name="weight")
```
- - -
### Linux
directory를 지우다가 Input/output error로 삭제가 안될 때가 있다.           
```
$rm -rf directory/
rm: cannot remove `directory': Input/output error
```
차분하게 막힌 곳을 뚫어서 해결한다.
```
$cd directory/
$losf +D .
COMMAND     PID    USER   FD   TYPE DEVICE  SIZE/OFF  NODE NAME
bash    4022366 jdj0715  cwd    DIR  0,114        49 69113 .
hadd    4102206 jdj0715    5r   REG  0,114 490708992 77997 ./.fuse_hidden000130ad00000001
lsof    4105633 jdj0715  cwd    DIR  0,114        49 69113 .
lsof    4105634 jdj0715  cwd    DIR  0,114        49 69113 .
$kill -9 4102206
$rm -rf directory/
```
- - -
### dlib install on OS X
맥 환경에서 dlib 설치할 때,
```
pip install dlib

Collecting dlib
  Using cached https://files.pythonhosted.org/packages/05/57/e8a8caa3c89a27f80bc78da39c423e2553f482a3705adc619176a3a24b36/dlib-19.17.0.tar.gz
Building wheels for collected packages: dlib
  Building wheel for dlib (setup.py) ... error
           ...
           
    RuntimeError:
    *******************************************************************
     CMake must be installed to build the following extensions: dlib
    *******************************************************************
    
    
    ----------------------------------------
Command "/Users/jeongdongjun/anaconda3/envs/ws/bin/python -u -c "import setuptools, tokenize;__file__='/private/var/folders/72/6zzm6mtx1mz13dhzq2w41rdw0000gn/T/pip-install-_gz5a81x/dlib/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /private/var/folders/72/6zzm6mtx1mz13dhzq2w41rdw0000gn/T/pip-record-6ute8pnc/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /private/var/folders/72/6zzm6mtx1mz13dhzq2w41rdw0000gn/T/pip-install-_gz5a81x/dlib/
```

Solution                
CMake를 설치해주고 다시 dlib을 설치한다.
```
$pip install cmake
$pip install dlib
```
- - -
### ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
```
img_to_array(load_img(fileName), dtype=uint)
ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
```

(1) re-install PIL
```
$pip install PIL
...
from PIL import Image
```
(2) upgrade some packages
```
$pip install --upgrade tensorflow keras numpy pandas sklearn pillow
```
- - -

### OpenCV installation error
In Ubuntu 18.04, OpenCV 4.1.0             

### make -j8
```
#include <hdf5.h> ^~~~~~~~ compilation terminated. modules/hdf/cmakefiles/opencv_hdf.dir/build.make:62: recipe for target 'modules/hdf/cmakefiles/opencv_hdf.dir/src/hdf5.cpp.o' failed make[2]: *** [modules/hdf/cmakefiles/opencv_hdf.dir/src/hdf5.cpp.o] error 1 cmakefiles/makefile2:3606: recipe for target 'modules/hdf/cmakefiles/opencv_hdf.dir/all' failed make[1]: *** [modules/hdf/cmakefiles/opencv_hdf.dir/all] error 2 make[1]: *** waiting for unfinished jobs.... [ 30%] built target opencv_ml [ 30%] built target opencv_surface_matching [ 33%] built target opencv_imgproc makefile:162: recipe for target 'all' failed
```

```
collect2: error: ld returned 1 exit status modules/core/cmakefiles/opencv_test_core.dir/build.make:1092: recipe for target 'bin/opencv_test_core' failed make[2]: *** [bin/opencv_test_core] error 1 cmakefiles/makefile2:2758: recipe for target 'modules/core/cmakefiles/opencv_test_core.dir/all' failed make[1]: *** [modules/core/cmakefiles/opencv_test_core.dir/all] error 2 makefile:162: recipe for target 'all' failed
```

solutions               
Remove the build directory and re-try to follow it.               
If you are anaconda user, work in conda environment instead of workon virtualenv.              

```
conda activate your_workspace
cd ~/Desktop/opencv
rm -rf build
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D OPENCV_GENERATE_PKGCONFIG=YES ..
      
make -j8

# go to 100 %
sudo make install
```

To check the c++ opencv file,
```
vi main.cpp

#include "opencv.hpp"
 
using namespace cv;
using namespace std;
 
int main(int argc, char** argv) {
  cout << "OpenCV version : " << CV_VERSION << endl;
}

g++ main.cpp `pkg-config --libs --cflags opencv4`
./a.out
```
- - -

### opencv installation
If you can not import the opencv, after install the opencv even in the conda environment.             
This works to me.
```
conda install -c conda-forge opencv
```

- - -

### opencv with yolo or other programs
```
(yolo) ubuntu@nipa2020-0987:~/djplace/darknet$ make
chmod +x *.sh
g++ -std=c++11 -std=c++11 -Iinclude/ -I3rdparty/stb/include -DOPENCV `pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv
` -DGPU -I/usr/local/cuda/include/ -DCUDNN -Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC -Ofast -DOPENCV -DGPU -DCUDNN 
-I/usr/local/cudnn/include -c ./src/image_opencv.cpp -o obj/image_opencv.o
Package opencv was not found in the pkg-config search path.
Perhaps you should add the directory containing `opencv.pc'
to the PKG_CONFIG_PATH environment variable
No package 'opencv' found
```

It means you don't have proper opencv in the path of /usr/local/include/$here.    
so we need to install the libopencv*
```
sudo apt-get update
sudo apt-get install libopencv*
```

then make again.

- - -

### PyTorch, gpu problem
If you meet some problems about GPU using pyTorch,    
you need to install pytorch again.
```
python
import torch
torch.cuda.is_available()
False
```

go to here official pytorch website,
<https://pytorch.org/get-started/locally/>    
and check the proper options.    
```
# for example
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

- - -

### jupyter notebbok / lab in linux server   
원격 서버에서 주피터를 사용하기 위한 세팅.    
```
ipython
from IPython.lib import passwd
passwd()
#copy your SHA key value
```

```
jupyter notebook --config 
vi .jupyter/jupyter_notebook_config.py
```
    
```
c = get_config()
c.NotebookApp.ip = "$.$.$.$" # your server ip, ifconfig
c.NotebookApp.password = '$$$$' # paste your SHA key value on here
c.NotebookApp.open_browser = False
```

```
jupyter notebook --ip=$.$.$.$ --no-browser
```

- - -

### import tqdm
tqdm을 import할 때 생기는 문제    
```
import tqdm
tqdm(total=int(length))
```
TypeError: 'module' object is not callable

import 구문을 수정해서 쉽게 해결할 수 있다.    
```
from tqdm import tqdm
```

- - -

### import ray
ray를 import할 때 생기는 문제    

It could be caused by the python version,    
you should check out the range for using the ray.    
Do downgrade your python.    
```
pip install ray
```
ERROR: Could not find a version that satisfies the requirement ray ERROR: No matching distribution found for ray
   
```
conda install python=3.8
pip install ray
```

- - -

### pandas 값 바꾸기
pandas file의 값을 바꿀 때 생기는 워닝 메시지    
 
```
file['blue_kda_mid'][17] = "['010/1']"

<ipython-input-80-256c6f241f8e>:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  file['blue_kda_mid'][17] = "['010/1']"

```   

loc를 사용하자.    

```
file.loc[17, 'blue_kda_mid'] = "['010/5']"
```
      
- - -


### GPU memory 안빠질 때
```
(1) nvidia-smi 로 확인    
(2) kill -9 PID 로 끄기

or

(1) sudo fuser -v /dev/nvidia*
(2) kill -9 PID 로 끄기
```
      
- - -
      
### RuntimeError
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
- GPU code running error    
```
Traceback (most recent call last):
  File "/home/june/analysis-lol-minimap-yolov5/train.py", line 602, in <module>
    main(opt)
  File "/home/june/analysis-lol-minimap-yolov5/train.py", line 500, in main
    train(opt.hyp, opt, device)
  File "/home/june/analysis-lol-minimap-yolov5/train.py", line 317, in train
    pred = model(imgs)  # forward
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/june/analysis-lol-minimap-yolov5/models/yolo.py", line 123, in forward
    return self.forward_once(x, profile, visualize)  # single-scale inference, train
  File "/home/june/analysis-lol-minimap-yolov5/models/yolo.py", line 155, in forward_once
    x = m(x)  # run
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/june/analysis-lol-minimap-yolov5/models/common.py", line 136, in forward
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/june/analysis-lol-minimap-yolov5/models/common.py", line 44, in forward
    return self.act(self.bn(self.conv(x)))
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 423, in forward
    return self._conv_forward(input, self.weight)
  File "/home/june/anaconda3/envs/ws/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 419, in _conv_forward
    return F.conv2d(input, weight, self.bias, self.stride,
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution      
```

Batch size를 줄여주거나, VRAM 사용량을 줄여야 한다.    

- - -


### MAC에서 분할압축파일 해제해야할 때 z01 zip
다른 방법 다 해도 안됨.
```
      unable to exapand "$.zip". it is an unsupported format.
```
appstore에서 the unarchiver 다운로드    
      
- - -


### 많은 파일들을 cp해야할 때
```
      cp ~
      -bash: /usr/bin/cp: Argument list too long
```

```
      find ./from_directory/ -name "*" -exec cp {} ./to_directory/ \; -print
```

- - -


### 많은 파일들을 wc -l 해야할 때
```
      ls ~ | wc -l
      -bash: /usr/bin/ls: Argument list too long
```

```
      find 2021_data_boucher_qualified -type f -name '*.png'  | wc -l
```
- - -


### 많은 파일들을 rm 해야할 때
```
      rm ~
      -bash: /usr/bin/rm: Argument list too long
```

```
      find ./from_directory/ -name '.*.png' -type f -delete
```

      
- - -


### Docker issue
```
$ docker run hello-world
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.

```
https://github.com/occidere/TIL/issues/116 해당 이슈를 참고한다.    
https://github.com/occidere/TIL/issues/116#issuecomment-745111682    

- - -

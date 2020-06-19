---
title: "Machine Learning Set Up"
date: 2019-04-10 20:59:00 +0900
categories: Machine Learning
---

First of all, you should think of compatiblity about your environment.    


Check the compatible versions within your environment and programs.    
(1) TensorFlow $.$ or PyTorch $.$    
(2) Consider your GPU architecture    
(3) Match your software versions about GPU, CUDA, cuDNN by cuDNN support matrix.    
<https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html>    

For installation of NVIDIA GPU environment, order is graphic driver > CUDA > cuDNN.    

## Installation NVIDIA graphic driver    
(1) Check your GPU hardware information and get drivers recommended    
```
# ubuntu-drivers devices

(base) dongjun@dongjun-System-Product-Name:~$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd000017C2sv00003842sd00002990bc03sc00i00
vendor   : NVIDIA Corporation
model    : GM200 [GeForce GTX TITAN X]
driver   : nvidia-driver-390 - distro non-free
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-410 - third-party free
driver   : nvidia-driver-440 - third-party free recommended
driver   : nvidia-driver-415 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

(2) or remove your previous graphic driver (if you need)    
```
sudo apt --purge autoremove nvidia*
```

(3) Installation NVIDIA graphic driver
```
sudo add-apt-repository ppa:graphics-drivers/ppa   
sudo apt update
#sudo apt-get install nvidia-driver-$$$ ($$$ is recommended number from ubuntu-drivers devices)
sudo apt-get install nvidia-driver-440 # in my case, recommended number was 440
sudo reboot

# check about your graphic information by GUI
nvidia-settings
```


## Installation CUDA    
NVIDIA 웹페이지에서 GPU의 architecture에 맞는 CUDA too-kit을 다운로드합니다.     
(1) Download the proper CUDA too-kit for your GPU, on NVIDIA webpage.    
<https://developer.nvidia.com/cuda-toolkit-archive>    

Check your options for downloading CUDA and get that dev file.    

for CUDA 10.1,    
<https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal>

Maybe it is on your Downloads directory now, /home/your_name/Downloads        

(2) or remove your previous CUDA (if you need) 
You first check your /usr/local directory or around there.    
```
cd /usr/local/
# and maybe there are cuda, cuda-$.$ directories.

(base) dongjun@dongjun-System-Product-Name:~$ cd /usr/local/
(base) dongjun@dongjun-System-Product-Name:/usr/local$ ls
bin   etc        include  libexec  sbin   src
cuda  cuda-10.1  games    lib      man    share  var
```

And remove them if you had it already.    
```
sudo apt-get --purge -y remove 'cuda*'
sudo apt remove --autoremove nvidia-cuda-toolkit

sudo apt-get autoremove --purge cuda
sudo rm /etc/apt/sources.list.d/cuda*
```
Then you can check that there is no CUDA directory now, on /usr/local/CUDA    

(3) Installation the CUDA tool-kit    
(3-1) Let's move on for CUDA tool-kit dev file.    
in EULA, you can easily accept it if you press ctrl+c and then.     
```
cd Downloads
# or elsewhere about CUDA tool-kit dev file

# depackage the CUDA dev file
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

# add some lines into ~/.bashrc file
sudo gedit ~/.bashrc
# or sudo vi ~/.bashrc

...
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```

(3-2) Set the cuda ppa
```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

(3-3) Installation our CUDA    
```
sudo apt update
sudo apt install cuda-10-1

echo 'export PATH=/usr/local/cuda-10.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc

ldconfig  

sudo reboot
```


You can check your graphic information also CUDA.       
```
# nvcc -V

(base) dongjun@dongjun-System-Product-Name:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Fri_Feb__8_19:08:17_PST_2019
Cuda compilation tools, release 10.1, V10.1.105

```

```
(설치전) (before installation CUDA)
(base) dongjun@dongjun-System-Product-Name:~$ nvidia-smi
-bash: nvidia-smi: command not found
```

```
(설치후) (after installation CUDA)
(base) dongjun@dongjun-System-Product-Name:~$ nvidia-smi
Fri Jun 19 10:58:50 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 00000000:01:00.0 Off |                  N/A |
| 25%   65C    P0    76W / 250W |    927MiB / 12212MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1294      G   /usr/lib/xorg/Xorg                           395MiB |
|    0      1560      G   /usr/bin/gnome-shell                         267MiB |
|    0      2378      G   ...AAAAAAAAAAAAAAgAAAAAAAAA --shared-files   254MiB |
+-----------------------------------------------------------------------------+
```
   
If you want to monitor that by real-time     
```
watch -n -d 0.5 nvidia-smi
```


## Installation cuDNN
NVIDIA 웹페이지에서 CUDA version에 맞는 proper cuDNN version을 다운로드합니다.     
(1) Download the proper cudNN for CUDA, on NVIDIA webpage.    
<https://developer.nvidia.com/rdp/cudnn-download>    

Agree the Terms of cuDNN License, then check your options for downloading cuDNN dev file.    
In my case, it was 'Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1'    
and 'cuDNN Runtime Library for Ubuntu18.04 (Deb)'    

Maybe it is on your Downloads directory now, /home/your_name/Downloads        

(2) Depackage the cuDNN library    
```
cd Downloads

sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
```

(3) Check out the result of cuDNN
```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

(base) dongjun@dongjun-System-Product-Name:~$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 5
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#include "driver_types.h"
```

ERROR: Symbolic link
```
cd /usr/local/cuda/lib64
ls -la libcudnn*

(base) dongjun@dongjun-System-Product-Name:/usr/local/cuda/lib64$ ls -la libcudnn*
-rwxr-xr-x 1 root root 428711256 Jun 11 11:21 libcudnn.so
-rwxr-xr-x 1 root root 428711256 Jun 11 11:21 libcudnn.so.7
-rwxr-xr-x 1 root root 428711256 Jun 11 11:21 libcudnn.so.7.6.5
-rw-r--r-- 1 root root 403829728 Jun 11 11:21 libcudnn_static.a
```
You can see there is no link within libcudnn things,    
you should make them linked like symbolic.    

```
(base) dongjun@dongjun-System-Product-Name:/usr/local/cuda/lib64$ sudo ln -sf libcudnn.so.7.6.5 libcudnn.so.7
(base) dongjun@dongjun-System-Product-Name:/usr/local/cuda/lib64$ sudo ln -sf libcudnn.so.7 libcudnn.so
(base) dongjun@dongjun-System-Product-Name:/usr/local/cuda/lib64$ ls -al libcudnn*
lrwxrwxrwx 1 root root        13 Jun 19 12:26 libcudnn.so -> libcudnn.so.7
lrwxrwxrwx 1 root root        17 Jun 19 12:26 libcudnn.so.7 -> libcudnn.so.7.6.5
-rwxr-xr-x 1 root root 428711256 Jun 11 11:21 libcudnn.so.7.6.5
-rw-r--r-- 1 root root 403829728 Jun 11 11:21 libcudnn_static.a
```


### Anaconda
1. 아나콘다 홈페이지에 들어가서 다운로드한다.   
(1) click -> <https://www.anaconda.com/distribution/>

(2) 아나콘다 홈페이지에서 Download 버튼을 누른다.

(3) 운영체제를 선택한다. (Windows / macOS / Lunux)

(4) Python 3.7 version의 Download 버튼을 누른다.

(5) Anaconda 패키지를 푼다.

(6) 아나콘다 설치가 완료되었는지 확인한다.


2. 아나콘다 아카이브: <https://repo.continuum.io/archive/> 
Python+OS를 알맞게 고려하여, 원하는 Anaconda 설치가능.


3. 초간단, Linux Server에서 내 워크스페이스에 설치할 때
```
$wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
$sh Anaconda3-5.3.1-Linux-x86_64.sh
```

설치 경로는 /home/your_username/anaconda3 가 적당.

맨마지막에 MS 스폰서 광고 들어있네요...
```
...
For this change to become active, you have to open a new terminal.

Thank you for installing Anaconda3!

===========================================================================

Anaconda is partnered with Microsoft! Microsoft VSCode is a streamlined
code editor with support for development operations like debugging, task
running and version control.

To install Visual Studio Code, you will need:
  - Administrator Privileges
  - Internet connectivity

Visual Studio Code License: https://code.visualstudio.com/license

Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]
>>> Please answer 'yes' or 'no':
>>> no
```

그리고, PATH 경로 잡아주면 완료
```
$source ~/.bashrc
```

```
(base) -MacBook-Pro:~ $conda --version
conda 4.6.11
(base) -MacBook-Pro:~ $python --version
Python 3.7.3
```

필요하면, 아나콘다를 최신버전으로 업데이트한다.
```
(base) -MacBook-Pro:~ $conda update -n base conda
```

아나콘다를 사용할 가상환경의 이름을 지어준다.
```
(base) -MacBook-Pro:~ $conda create -n 가상환경이름 python=3.7 anaconda
(base) -MacBook-Pro:~ $source activate 가상환경이름

#Ex) 가상환경의 이름을 ws라고 짓는다면,
(base) -MacBook-Pro:~ $conda create -n ws python=3.7 anaconda
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda deactivate
(base) -MacBook-Pro:~ $ 
```

아나콘다 가상환경 지우기
```
$conda remove --name 가상환경이름 --all
```

---

### Jupyter
아나콘다 설치를 완료하였으면, 사실 Jupyter를 사용할 수는 있다.   
하지만, jupyter notebook의 원활한 세팅을 위하여 Jupyter를 설치한다.   
설치 과정은 아나콘다의 가상환경 안에서 진행한다.
```
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda install jupyter notebook
(ws) -MacBook-Pro:~ $jupyter noteobook

#여러가지 옵션 설정을 위한 config.py 생성.
(ws) -MacBook-Pro:~ $jupyter noteobook --generate-config
Writing default config to: /Users/jeongdongjun/.jupyter/jupyter_notebook_config.py

#기본경로를 설정해준다.
(ws) -MacBook-Pro:~ $vi /Users/jeongdongjun/.jupyter/jupyter_notebook_config.py
(jupyter_notebook_config.py 안에서 #c.NotebookApp.notebook_dir = '' 라고 쓰여져있는 라인을 찾아준다.)
...
## The directory to use for notebooks and kernels.
#c.NotebookApp.notebook_dir = ''
...
#이렇게 되어있을 텐데, 앞에 #을 지우고 ''안에다가 기본경로를 적어준다. 이렇게,
...
c.NotebookApp.notebook_dir = '/Users/jeongdongjun/work'
...
```
끝.

---

### Tensorflow
아나콘다 가상환경 안에서 Tensorflow를 설치한다.
```
#Install the tensorflow
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda install tensforflow
```

Tensorflow의 설치를 확인한다.
```
(ws) -MacBook-Pro:~ $python
Python 3.7.3 (default, Mar 27 2019, 16:54:48) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.__version__
'1.13.1'
```
---

### Keras
Tensorflow를 설치할 때와 똑같다.
```
#Install the tensorflow
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda install keras
(ws) -MacBook-Pro:~ $python
Python 3.7.3 (default, Mar 27 2019, 16:54:48) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import keras
Using TensorFlow backend.
```
끝.

---

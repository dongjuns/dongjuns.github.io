---
title: "Machine Learning Set Up"
date: 2019-04-10 20:59:00 +0900
categories: Machine Learning
---

### GPU   
(1) GPU 모델 및 정보 확인 + get drivers recommended   
```
ubuntu-drivers devices
```
(2) GPU installation
```
# 이전 그래픽 드라이버를 지워야 한다면,
#sudo apt --purge autoremove nvidia*

# 설치과정
sudo add-apt-repository ppa:graphics-drivers/ppa   
sudo apt update
sudo apt-get install nvidia-driver-### (recommended number)
sudo reboot

# 설치확인
nvidia-settings
```

### CUDA   
nvidia 웹페이지에서 GPU의 architecture에 맞는 CUDA tool-kit을 다운로드합니다.     
<https://developer.nvidia.com/cuda-toolkit-archive>

아마 Downloads 디렉토리에 받아져있을텐데, CUDA를 설치합니다.

정상적으로 설치가 되었다면, nvidia-smi 명령어로 확인가능합니다.
```
(설치전)
[dojeong@gate analysis]$ nvidia-smi
-bash: nvidia-smi: command not found
```

```
(설치후)
[dojeong@gate2 analysis]$ nvidia-smi
Mon Jul 29 15:30:06 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:04:00.0 Off |                    0 |
| N/A   33C    P0    28W / 250W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:08:00.0 Off |                    0 |
| N/A   32C    P0    24W / 250W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  Off  | 00000000:0C:00.0 Off |                    0 |
| N/A   34C    P0    26W / 250W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  Off  | 00000000:0F:00.0 Off |                    0 |
| N/A   32C    P0    26W / 250W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

You can check the nvidia graphic driver & CUDA Version.      
If you want to monitor that on real-time,     
```
watch -n -d 0.5 nvidia-smi
```

And If there is a big zombie job to die,      
```
top & kill -9 pid
```

---
(1) cuda 찌꺼기 삭제   
```
cd /usr/local/cuda/ or cudax.x
```

```
sudo apt-get --purge -y remove 'cuda*'
sudo apt remove --autoremove nvidia-cuda-toolkit

sudo apt-get autoremove --purge cuda
sudo rm /etc/apt/sources.list.d/cuda*
```

(#2) Download the run file at the NVIDIA cuda   

```
sudo sh filename.run

ctrl+c and accept more more
```

```
#reboot and check
sudo reboot
...
nvcc -V
```


### for 10.1
<https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal>

download the dev file,
```
cd Downloads

sudo dpkg -i cuda-repo-ubuntu...deb
sudo apt-key add /var/cuda-.../7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

sudo vi ~/.bashrc

#add these
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```



(2) set the cuda ppa
```
sudo apt update

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

(3) cuda 설치
```
sudo apt update

sudo apt install cuda-10-1

echo 'export PATH=/usr/local/cuda-10.1/bin:$PATH' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc

ldconfig  
```


<https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal>   
```

```


(4) cuDNN 설치   

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

설치 경로는 /home/username/anaconda3 정도가 적당.



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

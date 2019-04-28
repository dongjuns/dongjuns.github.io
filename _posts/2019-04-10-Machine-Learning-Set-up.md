---
title: "Machine Learning Set Up"
date: 2019-04-10 20:59:00 +0900
categories: Machine Learning
---

### Anaconda
아나콘다 홈페이지에 들어가서 다운로드한다.   
(1) click -> <https://www.anaconda.com/distribution/>

(2) 아나콘다 홈페이지에서 Download 버튼을 누른다.

(3) 운영체제를 선택한다. (Windows / macOS / Lunux)

(4) Python 3.7 version의 Download 버튼을 누른다.

(5) Anaconda 패키지를 푼다.

(6) 아나콘다 설치가 완료되었는지 확인한다.

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
끝.


이어서,

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

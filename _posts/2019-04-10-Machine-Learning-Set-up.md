---
title: "Machine Learning Set Up"
date: 2019-04-10 20:59:00 +0900
categories: Machine Learning
---

#Anaconda
go to -> https://www.anaconda.com/distribution/

click that box which 'Download'

and Select your OS,

click the Download button for Python 3.7 version

Double-click the pkg file to unzip.

and have a version check,
```
(base) -MacBook-Pro:~ $conda --version
conda 4.6.11
(base) -MacBook-Pro:~ $python --version
Python 3.7.3
```

If we need, can update the anaconda to current version.
```
(base) -MacBook-Pro:~ $conda update -n base conda
```
then, make a nickname for starting conda.

```
(base) -MacBook-Pro:~ $conda create -n 하고싶은이름 python=3.7 anaconda
(base) -MacBook-Pro:~ $source activate 하고싶은이름

#Ex)
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda deactivate
(base) -MacBook-Pro:~ $ 
```
That is all.


이어서,

#Jupyter

If you follow this step, it means that you already have a jupyter.

In the Anaconda work space,
```
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $jupyter noteobook

#to set a default starting path
(ws) -MacBook-Pro:~ $jupyter noteobook --generate-config
Writing default config to: /Users/jeongdongjun/.jupyter/jupyter_notebook_config.py

#set the default path.
(ws) -MacBook-Pro:~ $vi /Users/jeongdongjun/.jupyter/jupyter_notebook_config.py
(in the jupyter_notebook_config.py, find a #c.NotebookApp.notebook_dir = '' <- this line.)
...
## The directory to use for notebooks and kernels.
#c.NotebookApp.notebook_dir = ''
delete # and fill the form into ''
c.NotebookApp.notebook_dir = '/Users/jeongdongjun/work'
...
```
This will be your default path at jupyter notebook.



```
#many advices for using jupyter
(ws) -MacBook-Pro:~ $jupyter noteobook --help
```


#TensorFlow

In the Anaconda work space,
```
#Install the tensorflow
(base) -MacBook-Pro:~ $conda activate ws
(ws) -MacBook-Pro:~ $conda install tensforflow
```

and check the installation of tensorflow.
```
(ws) -MacBook-Pro:~ $python
Python 3.7.3 (default, Mar 27 2019, 16:54:48) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.__version__
'1.13.1'
```

super same to install the keras,
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
That is all.

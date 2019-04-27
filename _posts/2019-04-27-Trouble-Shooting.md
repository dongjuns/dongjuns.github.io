---
title: "Trouble Shooting"
date: 2019-04-27 21:19:00 +0900
categories: Trouble Shooting
---

#matplotlib
주피터 노트북을 사용할 때, matplotlib을 import하다가 에러가 날 때가 있다.
```
In [] import matplotlib.pyplot as plt
Users/jeongdongjun/anaconda3/lib/python3.7/site-packages/matplotlib/font_manager.py:232: UserWarning: 
Matplotlib is building the font cache using fc-list. This may take a moment.   
'Matplotlib is building the font cache using fc-list. '
```

이럴 때는
```
$cd ~/.matplotlib
$rm -rf tex.cache
```

#jupyter에서 tensorflow import 에러
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

```
In []import tensorflow as tf
In []tf.__version__
Out []'1.13.1'
```

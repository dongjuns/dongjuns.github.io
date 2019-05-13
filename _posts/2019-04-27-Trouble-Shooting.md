---
title: "Trouble Shooting"
date: 2019-04-27 21:19:00 +0900
categories: Trouble Shooting
---

### matplotlib import 에러
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

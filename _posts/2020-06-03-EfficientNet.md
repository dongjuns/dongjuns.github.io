---
title: "EfficientNet"
date: 2020-06-24 11:37:00 +0900
categories: simpledet EfficientNet
---


### For parameter calculation

```
# EfficientNetParameterCalculator.py
a = 1.2
b = 1.1
r = 1.15

a_list = []
b_list = []
r_list = []

phi_list = [0, 0.5, 1, 2, 3.5, 5, 6, 7] # phi value of 0, 1, 2, 3, 4, 5, 6, 7

for phi in phi_list:
    new_a = pow(a, phi)
    new_b = pow(b, phi)
    new_r = pow(r, phi)
    a_list.append(new_a)
    b_list.append(new_b)
    r_list.append(new_r)
```


For channel of each stage i
```
channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
new_channels = []

b_length = len(b_list)
c_length = len(channels)

for b_num in range(b_length):
    for c_num in range(c_length):
        new_channel = b_list[b_num] * channels[c_num]
        new_channels.append(new_channel)
```

For depth of each stage i
```
layers = [1, 1, 2, 2, 3, 3, 4, 1, 1]
new_layers = []

a_length = len(a_list)
l_length = len(layers)

for a_num in range(a_length):
    for l_num in range(l_length):
        new_layer = a_list[a_num] * layers[l_num]
        new_layers.append(new_layer)
```

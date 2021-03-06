---
title: "Image Classification"
date: 2019-04-28 20:48:00 +0900
categories: Machine Learning
---

This is recipe of image classification, also contains general knowledge about techniques.     
이미지 인식에 대한 딥러닝 레시피입니다. 특정 기술들에 대한 설명도 포함합니다.

### 0. Import Libraries
I think in the image classification, there are special libraries which have almost always used.     
Pandas를 이용하여 data를 load하고, numpy로 간단한 linear algebra들을 사용하며, seaborn과 matplotlib으로 visualization을 해줍니다.     
```
# General Libraries
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
```

scikit-learn, tensorflow, keras, pytorch 등등 여러가지 framework을 이용하여 Machine Learning과 Deep Learning을 할 수 있습니다.     
정답은 없으며, 개인적인 생각으로는 keras가 building model이 매우 쉽습니다. pytorch는 python coding에 있어서 굉장히 친숙하게 느껴집니다.     
필요한 라이브러리가 있다면, jupyter notebook 환경안에서 간단하게 설치합니다.
```
!pip install nameOfLibrary
```

```
import something framework, anything...

# scikit-learn style
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# keras style
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten,
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# tensorflow style
import tensorflow as tf

# pytorch style
import torch.nn as nn
```


### 1. Load dataset
Pandas를 이용하여 손쉽게 Load할 수 있습니다. 또는 computer vision을 이용하여 data를 사용할 수 있습니다.
```
# PANDAS style, import pandas as pd
# .csv
dataset = pd.read_csv("path/dataset.csv", axis=1)

# .txt
dataset = pd.read_csv("path/dataset.txt", usecols=())
```

```
# Computer Vision style, import cv2
def loadData():
    result = []
    datasets = ['trainPath', 'testPath']
   
    for dataset in datasets:
        directory = dataset
        images = []
        labels = []
        
        for folder in os.listdir(directory):
            currentLabel = folder
       
            for file in os.listdir(directory + "/" + folder):
                imgPath = directory + "/" + folder + "/" + file
                currentImg = cv2.imread(imgPath)
                currentImg = cv2.resize(currentImg, (120, 120))
                images.append(currentImg)
                labels.append(currentLabel)
                
        images, labels = shuffle(images, labels)
        result.append((images, labels))
            
    return result
```
```
(trainImages, trainLabels), (testImages, testLabels) = loadData()    
```


### 2. Visualize and Explore the images
Visualization can give you a insight, we don't need to know about all of method to visualize,     
but there are also specifically good techniques.
Presentation이나 Discussion을 위하여 data를 시각화할 수 있어야 합니다. 용도에 맞게 가장 간단하고, 효과적인 방법 몇가지를 소개합니다.

(1) univariate analysis, i.e, target variable.      
단독변수 분석을 할 때 입니다. 종속변수를 예를 들자면, 종속변수의 class 분포, class별로 몇개씩 있는지가 중요할 수 있습니다.
```
# count plot
sns.countplot(variable) # Draw a bar plot of each class of the feature.
variable.value_counts() # This show you a number of each class.

# distribution plot

# pie plot, in case of the categorical feature
classes = np.bincount(train)
plt.pie(classes, labels=['className1','className2','className3'], autopct='%1.2f%%')
plt.title('The title whatever you want')
plt.show()
```

때때로 image가 어떻게 생겼는지 직접 확인하고 싶을 때도 있습니다.
```
# 1 image
idx = np.random.randint(trainImages.shape[0])
plt.figure()
plt.imshow(trainImages[idx])
plt.title('This image is #{} '.format(trainLabels[idx]))
plt.show()
```

```
# many images
fig = plt.figure(figsize=(10,10))
fig.suptitle('Sample images of the dataset', fontsize=12)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainImages[i])
    plt.xlabel(train_labels[i])
plt.show()
```

### 3. Preprocessing images
Scaling

### 4. Modeling

### 5. Image Data Augmentation

### 6. Hyper-parameter optimization

### 7. Classify the images

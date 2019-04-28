---
title: "Keras Recipe"
date: 2019-04-28 20:48:00 +0900
categories: Machine Learning
---

#### Fully Connected Network

### Import
import keras

### Datasets
load datasets
(1) train&test features : reshape & astype
(2) labels : to_categorical

### Sequential
model = models.Sequential()

### Layers
model.add(layers.Dense($$$, activation='', input_shape=($$$, )))
model.add(layers.Dense($$$, activation='softmax'))

### Compile
model.compile(optimizer='', loss='', metrics=[''])

### Training
model.fit(trainFeatures, trainLabels, batch_size=$$$, epochs=$$$)

### Evaluate or History
testLoss, testAccuracy = model.evaluate(testFeatures, testLabels)
print('Evaluate the model with loss and accuracy :', testLoss, testAccuracy)

hist = model.fit(trainFeatures, trainLabels, batch_size=$$$, epochs=$$$, validation_data=(valFeature, valLabels))
print(hist.history['loss'])
print(hist.history['val_loss'])
print(hist.history['acc'])
print(hist.history['val_acc'])



#### CNN, Convolution Neural Network
### Import
import keras

### Datasets
load datasets
(1) train&test features : reshape & astype
(2) labels : to_categorical

### Sequential
model = models.Sequential()

### Layers
model.add(layers.Conv2D($$$,($$$, $$$), activation='', input_shape=($$, $$, $)))
model.add(layers.MaxPooling2D(($, $)))
model.add(layers.Conv2D($$, ($, $), activation=''))
model.add(layers.MaxPooling2D(($, $)))
model.add(layers.Conv2D($$, ($, $), activation=''))
model.add(layers.Flatten())
model.add(layers.Dense($$, activation=''))
model.add(layers.Dense($$, activation=''))

### Compile
model.compile(optimizer='', loss='', metrics=[''])
              
### Training
model.fit(train_features, train_labels, batch_size=$$$, epochs=$$$)

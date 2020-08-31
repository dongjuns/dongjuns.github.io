---
title: "New Model Idea"
date: 2020-08-11 14:48:00 +0900
categories: classification detection modeling
---

1.CircleNet: Input -> Output classification    

2.Reverse DL: Output -> Input classification    

3.Reverse Auto Encoder: Output -> Input -> Output    
VS Auto Encoder: Input -> Encode -> Input    

4.Multiple Optimizer: Optimize but start at multiple points, this idea is based on that there is a good way to go on peak easily than other.    

5.How many dataset we need? calculator    

6.How many dataset we need only for GAN    

7.Dataset Similarity Calculation for different dataset    

8.From Baby To Child Net: Train from R, G, B... Circle, Triangle, Rectangle then classification.    

9.Dead pixel, Hot Pixel, Stuck Pixel augmentation    

10.Color Blind Net: Make Deuteranomaly, Protanopia, Tritanopia style augmentation and Test with normal vision

11.TripleNet

12.Tapetum Net: like cat, use the reflection thing.

13.Train 1 network for 1 class, and ensemble that with detection threshold VS 1 network for multiple classes

14.Multi-layer Auto Encoder: AE but there are much more layers, from input to latent space and from output to latent space.
AE: input-(Encode+Decode)-output, MAE: input-encode layer1-encode layer2-embed-decode layer2-decode layer1-output    

15.Sustainable Model: similar with online learning but also similar with self-supervised learning.    

16.Domain randomization for limited situation    

17.Dataset augmentation with calculating loss    

18.Classification then detection.    
One model but two model combined.    
Image classification -> detection if it exists, or not.    

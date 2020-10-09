---
title: "Importance of Image Format"
date: 2020-10-08 12:52:00 +0900
categories: Image Format 
---

Need to care about '3 color channel + 1 alpha channel' like as RGBA style,
because if you trained your model using 3 channel RGB images,    
reasonably it can not do inference whtin 4 channel images.    
It doesn't related only image file format with jpg, png, tiff, like that.    


### Good habit
if you meet some problems after finished your training the model, try to use other image.    
And think of the essence of the problem.    
maybe your image resolution, labeling, img type and image color channel.    
Extend your conscious.

---
title: "Importance of Image Format"
date: 2020-10-08 12:52:00 +0900
categories: Image Format 
---

물체에 반사된 빛은 카메라를 통해 2차원 데이터로 변환된다.    
카메라 내부의 이미지 센서를 통해 Red, Green, Blue 세가지 필터에 대하여 0~(2^n비트)-1 범위로 색상을 표현할 수 있고,    
일반적으로 하나의 픽셀에 8비트로 빛의 세기를 저장하기 때문에 R 8비트, G 8비트, B 8비트 총 24비트로 계산할 수 있다.    
저장하는 이미지의 해상도가 가로 1920, 세로 1280 일 때,    
한 장의 이미지는 1920 * 1280 * 24 bit 로 저장된다는 것을 알 수 있다.    
8비트가 1바이트이므로, 추가적인 단위 계산을 통하여 이미지의 바이트를 표현할 수 있다.    


Need to care about '3 color channel + 1 alpha channel' like as RGBA style,
because if you trained your model using 3 channel RGB images,    
reasonably it can not do inference whtin 4 channel images.    
It doesn't related only image file format with jpg, png, tiff, like that.    


### Good habit
if you meet some problems after finished your training the model, try to use other image.    
And think of the essence of the problem.    
maybe your image resolution, labeling, img type and image color channel.    
Extend your conscious.

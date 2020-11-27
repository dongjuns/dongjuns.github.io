---
title: "LOL Analysis Sketch"
date: 2020-11-27 18:11:00 +0900
categories: Deep_learning Data_Analysis
---


1.데이터셋을 어떻게 구축할 것인가? dataset
LOL 게임 영상에서, 특정 부분들을 분류하고 데이터를 축적할 수 있어야 함.    
OpenCV, 미니맵 부분에 대한 좌표를 입력하고 영상을 저장한다.    

-> 미니맵 영상 데이터셋 구축에 대한 알고리즘 구현 -> 추후에 자동화까지.    


OCR에 대한 필요성, LOL 영상 내에 팝업 문구를 인식하고 텍스트로 받아올 수 있는 알고리즘 필요.

챔피언 얼굴 겹쳤을 때, 어떻게 해결할 것인가? occlusion

데이터셋 축적에 따른 모델 업데이트를 어떻게 자동화할 것인가?
Life-long learning, offline/online learning    


서비스 연계의 측면에서, 어떻게 모델을 서빙할 

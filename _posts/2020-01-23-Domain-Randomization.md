---
title: "Notes of Domain Randomization"
date: 2020-01-23 10:58:00 +0900
categories: Deep Learning
---

# Deep Learning & Data      
Neural Network를 구성하고 실제로 비즈니스에 사용하기 위해서는, Model을 학습시키기 위한 데이터셋을 확보해야 합니다.      
Kaggle이나 Neural Net comptetion이 아닌,     
특정 문제에 대해서 비즈니스적으로 접근하고 Solution 으로 Deep Learning을 사용하려고 한다면,      
'데이터를 어떻게 모을까?' 라는 고민을 해결해야 합니다.      
마찬가지로, 이미 모델을 구축한 경우에도 모델의 Generalization을 확보하기 위해서 데이터가 더 필요하고,      
모델의 성능을 향상시키기 위해서도 데이터가 더 필요합니다.      
데이터의 부재에 대한 문제를 해결하고자 하는 여러가지 방법들 중에서,      
Few-Shot Learning과 Domaion Randomization에 대한 Paper들을 읽어보고 새로운 Idea를 도출해봅니다.     

- - -

## Few-Shot Learning

- - -

## Domain Randomization

There is a problem to have a lot of dataset to train a Deep Neural Network models.      
In robotics area, they can make tons of simulated images using image maker tools,     
And can have simulated physics simulator.     
Between simulated robotics (Simulation) and experiment on hardware (Real world), there is 'reality gap'.      
So, if you want to train your model using physics simulator, you need to break the gap.      
Simulation은 데이터의 생산성과 품질을 높일 수 있고, 이는 좋은 연구 결과의 도출로 이어집니다.      
하지만, 시뮬레이션과 실제실험 사이에 차이가 있음을 의미하는 Reality Gap이 동반되므로,
Improved data avaliablity를 이용하여 Reality Gap을 메꾸는 것이 아주 중요합니다.     

### Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World       
(20 Mar 2017) <https://arxiv.org/pdf/1703.06907.pdf>      
    
Abstract      
Reality gap: Separates simulated robotics from real-experiment on hardware.     
Domain Randomization: For training models on simulated images that transfer to real images by randomizing rendering in the simulator.     
Real image를 랜덤화 & 렌더링해서 simulated images를 만들고, 그것들을 이용하여 모델을 학습시키는 방법입니다.     
충분한 Domain Randomizaing을 통해서 이미지셋에 변동성을 부여하게 되면, real image도 variations의 하나로 여겨집니다.     
Simulated RGB images를 이용하여 object localization 모델을 학습시키고, real world의 실험 상황에서 object를 정확하게 잡아냈습니다.      

1. Introduction     
Physics Simulator는 robotics learning에 대해 도움을 줄 수 있습니다. 빠르고, 확장하기 쉽고, 데이터를 얻는 것 또한 저렴하고 쉽습니다.      
하지만, 시뮬레이션과 현실 사이에 생기는 차이를 줄여야 하는데요, 현실의 physical system과 똑같은 물리적 특성을 simulatior에도 적용해주는 System identification 방법은 아주 어렵고, 시간도 상당히 오래걸리고 오류도 자주 발생합니다. 아무리 Simulator를 잘 만들어도 현실적으로 unmodeled physics effect가 존재하며, 몇몇 image renderer들의 경우에는 오브젝트의 특징을 완벽하게 살릴 수가 없습니다.      
그래서, Reality gap을 줄이기 위한 방법으로 Domain randomization을 사용하고자 합니다. simulator를 randomization하고, model을 다양한 환경에 노출시키고 학습시키는 방법입니다. simulation 환경의 변동성이 충분히 크다면, simulation에서 학습된 model이 추가적인 작업없이도 real world까지 일반화할 수 있을 것이라는 hypothesis를 따릅니다. Low-fidelity simulated camera images를 이용할 것이고, Object localization을 위하여 model을 학습시킵니다. And we find that with a sufficient number of textures, pre-training the object detector using real images is unnecessary.

2. RELATED WORK     
 A. Object detection and pose estimation for robotics           
 Robotics 분야에서 object detection과 pose estimation은 수월하게 연구되고 있으며, 최근에는 3D model of object를 offiline construction하거나 learning하는 방법을 일반적으로 사용하고 있습니다. SIFT features를 이용하여 3D model과 match시키고, black-box nonlinear optimization algorithm을 사용하여 re-projection error of the SIFT points를 줄일 수 있습니다. Multiple camera frames를 이용한 방법, depth information을 이용한 방법, only monocular camera images를 이용한 방법이 성공적인 결과를 보였습니다. 이와 같은 전통적인 방법들과 비교해보면, 본 논문에서 사용하고 있는 방법의 경우에는 3D reconstruction과 같은 문제를 피할 수 있으고, 사용하기에도 간단하며 쉽다는 장점이 있습니다.            
 
 B. Domain adaptation           
 Computer Vision 연구자들은 source domain으로 학습시킨 vision-based model을 unseen target domain에 적용시키는 domain adaptation 방법에 대해서 연구해왔습니다. 다양한 접근 방법이 제안되었으며, 모델을 target domain으로 다시 학습시키는 방법, model의 weights를 source domain과 target domain의 통계적 특성을 고려하여 적용시키는 방법, domains의 invariant features를 학습시키는 방법, target domain으로부터 source domain을 mapping하고 그것을 learning하는 방법 등이 있습니다. 본 논문에서는 domain adaptation을 사용하지 않고, simulation to real world의 possibility of transfer를 연구합니다.             
 
 C. Bridging the reality gap            
 결국 요점은 'reality gap을 어떻게 메울 것인가?' 입니다. system identification, high-quality rendering과 같은 기술을 정교하게 적용하여 simulator를 real world와 아주 비슷하게 만들자는 방법이 있었고, 어느 정도의 성과를 거둘 수 있었습니다. 본 논문의 방법은 low-quality renderer와 적당한 수준으로 제작한 simulation environment를 이용하였습니다. Domain adaptation을 이용하는 방법도 있으며, 이 경우에는 simulation에서 학습시킨 controller를 tuning하는 것이 real world에서 처음부터 학습시키는 것보다 빨랐습니다. Simulated data를 이용하여 variational autoencoder를 학습시키고, real data를 이용하여 policy를 학습시킨 다음 feature of distribution을 mapping 시킴으로써 reality gap을 극복하였습니다. 이렇게 다양한 방법을 사용해서 bridging the reality gap을 하려는 시도들이 있었으며, domain randomization에 대한 결과도 있었으나, 본 논문이 갖고 있는 차이점은 카메라의 정확성이나 calibration 대신에 randomizing the position, orientation and field of view of the camera in the simulator를 이용하였다는 점입니다. 그리고 only non-realistic textures를 사용하였습니다.            


3. METHOD     

4. EXPERIMENTS      

5. CONCLUSION     

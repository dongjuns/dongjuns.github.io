---
title: "Computer Dictionary"
date: 2019-04-22 15:11:00 +0900
categories: Computer
---

### 운영체제 OS (Operating System)
Hardware & Software Operation Manager
컴퓨터 세계의 관리직. 자원을 할당해주고, 일을 시킨다.

32bit과 64bit의 차이.
bit의 갯수를 나타내고, 한번에 처리하는 데이터량을 의미한다.

```
32비트는 2^32 = 4,294,967,296
64비트는 2^64 = 18,446,744,073,709,55x,xxx 
```

32bit 에서는 4GB 넘는 메모리를 달아도 성능을 못 살림. (64bit는 2TB까지 사용가능.)

기존에는 32bit OS를 기반으로 프로그램을 만들어왔기 때문에, 64bit OS와 호환성 이슈가 있음.    
그렇기 때문에 처리속도는 64bit보다 느리지만, 프로그램들과 호환성이 좋고 안정적인 32bit OS를 사용.

### CPU (Central Processing unit)
중앙처리장치, 연산을 담당하는 코어로 이루어져있다.
코어가 여러개일때 -> 멀티코어.

Program(passive entity) > Process(active entity, 실행중인 프로그램) > Thread

1 Process = code + data + heap + stack(thread) 으로 이루어지며,
thread는 process 안에서 stack으로 존재하며, 여러가지 작업들을 순서대로 수행한다.    
1 Process를 multi thread 로 이용하면, 내부의 code&data&heap 를 공유해서 작업하기 때문에 빠르다.   
(thread-thread context switching 빠름)    
But, multi process 에 비해 접근성 및 구조가 까다롭고, 디버그가 어렵다.

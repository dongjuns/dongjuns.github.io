---
title: "LLM token efficiency"
date: 2025-11-05 12:35:00 +0900
categories: LLM, Token
---

When we use the LLM, the token is the standard cost unit. And we need to know the token process from the user's query to the LLM's generated response.    

Nowadays, most LLMs are multi-language model.    
But it doesn't mean the LLM can understand every language well.    
It definitely learn the multi-language sequences, so there would be the priority of the language, because the english data is much larger than other language data.    

This is the important point!    
If your question was in the training dataset, you can get the perfect result whether your question is much difficult or not.    


**MOHO**    
User query에서 모호한 부분들을 sLLM이 물어봐서 문장의 context를 충만하게 함.    
이중적인 단어가 쓰였거나, 데이터를 첨부안했는데 '내가 올린 데이터 분석해줘'와 같은 query가 들어왔다던가 하는 경우에는 MOHO score가 높게 나오도록.    
이런 경우, sLLM이 MOHO를 계산하고 추가 정보를 요청하는 형태로 작동하도록 함.    



**Query complete**
User query가 너무 짧거나 부족한 정보로 response를 요구할때.    
incomplete -> complete task로 볼 수 있을듯?
무엇을, 누구한테, 왜 사과하는지에 대한 내용이 없기 때문에 response의 완성도가 낮을 수 밖에 없음.    
예를 들어서, query: '사과하는 법'인 경우, 문맥적으로 사과하는 방법을 알려달라고 하는 것이겠지만,    



**TTP**    
single prompt에 너무 많은 요청을 넣으면 LLM이 제대로 응답못하는 경우 발생    
Let's think of step by step 과 같은 trick으로 해결하는 것이 아니라, prompt를 논리적으로 쪼개는 방식으로 해결하면 어떨까?    


**XLT**
비영어권 query를 넣을때 영어로 번역해서 넣어야함. high-resource language인 English의 도움을 최대화하기위해.    
근데, 질문 자체를 넣고 번역해서 알려달라고 하면 하나의 prompt에 여러 요청을 하는 것임.    

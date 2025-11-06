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


MOHO    
User query에서 모호한 부분들을 sLLM이 물어봐서 문장의 context를 충만하게 함.

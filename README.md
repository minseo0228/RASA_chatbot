# RASA_chatbot
## 2022 객체지향언어와 실습 Project
## 주제 : python 3.8, Rasa 3.1 기반 한국어 성폭력 피해자 진술 준거 챗봇
주어진 시나리오와 데이터셋을 이용
### RASA 챗봇의 기본 원리를 도식화한 그림입니다. 

<img width="500" alt="RASA" src="https://user-images.githubusercontent.com/103639821/207577706-5daf42f2-ca15-438f-924c-a69816a3f852.png" height="300">

1. [토크나이저 피처라이저 자료 및 정리본](https://www.notion.so/RASA-ba1ef38ff4554fe4b61724152acd2944)  

      
2. [협업 notion](https://www.notion.so/RASA-179c4128ea9d48cd80f947e40335d239)
<br/>

## 개발과정
![KakaoTalk_20221116_160812001](https://user-images.githubusercontent.com/103639821/207577911-767de489-3980-42c2-bfc8-0cbe9d04b0ed.png)
<br/>
## 목적 지향 대화 시스템(Task - Oriented Dialogue System)
    
    : 특정한 목적 또는 작업을 수행하는 것을 목표로 하며, 파이프라인 방식을 따른다.
    
    - 파이프라인 방식
        - 자연어 이해(NLU)
            - 사용자 발화의 영역(domain) & 의도(intent)를 파악해 사용자의 질문이나 응답에 어떤 내용(slot)이 나타났는지 파악  → 사전에 entity(챗봇이 필수적으로 파악해야 하는 개체)가 지정되어야 함.
        - 대화 관리(Dialogue management)
            - 이전의 대화 내용과 현재의 발화를 통해 대화 상태를 추적. 사용자가 대화의 흐름을 확인하고 다음 대화에서 확인할 내용을 결정하는 역할.
        - 자연어 생성(NLG)
            - 자연스러운 챗봇의 응답이나 질문을 생성하는 역할
            
## 토큰화 진행
<img width="423" alt="토큰화" src="https://user-images.githubusercontent.com/103639821/207588201-6fb7490f-e1af-40a8-8940-890be1ca5d43.png">
토큰화: konlpy - mecab형태소 분석기   
피처라이저의 경우 GLove로 사전에 임베딩된 데이터를 불러와 사용하였습니다. 

## 최종 결과
![챗봇](https://user-images.githubusercontent.com/103639821/207587761-e52e8455-527a-40d0-bcfc-84fa1a100bda.png)



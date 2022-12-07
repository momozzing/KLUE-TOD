# KLUE-TOD

## introduction 

[KLUE-wos dataset](https://huggingface.co/datasets/klue/viewer/wos/train)으로 만든 [KoGPT-2](https://huggingface.co/skt/kogpt2-base-v2) 기반의 간단한 한국어 목적 지향 대화 시스템입니다. 

GPT-2로 Multi-woz data를 이용한 SimpleTOD를 한국어로 구현하였습니다. 

KLUE-wos data는 dialogue act, DB가 없기 때문에 이를 제외하고 만들었습니다. 

![image](https://user-images.githubusercontent.com/60643542/206118327-890f119e-e31a-40b2-8c23-683b044d2f09.png)

![image](https://user-images.githubusercontent.com/60643542/206118836-9c50ef09-8101-4f67-be73-0704307fa3c4.png)

domain apdatation을 위한 special token도 유지하였습니다. 

![image](https://user-images.githubusercontent.com/60643542/206119028-ac3667e6-d721-4e14-a24e-5726fde2f1e2.png)


## How to run? 
```
1. git clone https://github.com/momozzing/KLUE-TOD.git

2. pip install -r requirements.txt

3. cd KLUE-TOD


```



## Reference

[Huggingface](https://huggingface.co/)

[KLUE-wos](https://huggingface.co/datasets/klue/viewer/wos/train)

[KoGPT-2](https://huggingface.co/skt/kogpt2-base-v2)

[SimpleTOD paper](https://arxiv.org/pdf/2005.00796.pdf)
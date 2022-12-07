# KLUE-TOD

## introduction 

[KLUE-wos dataset](https://huggingface.co/datasets/klue/viewer/wos/train)으로 만든 [KoGPT-2](https://huggingface.co/skt/kogpt2-base-v2) 기반의 간단한 한국어 목적 지향 대화 시스템입니다. 

GPT-2로 Multi-woz data를 이용한 SimpleTOD를 한국어로 구현하였습니다. 

KLUE-wos data는 dialogue act, DB가 없기 때문에 이를 제외하고 만들었습니다. 

![image](https://user-images.githubusercontent.com/60643542/206118327-890f119e-e31a-40b2-8c23-683b044d2f09.png)

![image](https://user-images.githubusercontent.com/60643542/206118836-9c50ef09-8101-4f67-be73-0704307fa3c4.png)

domain apdatation을 위한 special token을 사용하였습니다. 

![image](https://user-images.githubusercontent.com/60643542/206119028-ac3667e6-d721-4e14-a24e-5726fde2f1e2.png)


## How to train? 
```
1. git clone https://github.com/momozzing/KLUE-TOD.git

2. pip install -r requirements.txt

3. cd KLUE-TOD

4. sh train.sh 
```

## How to interactive? 
```
1. git clone https://github.com/momozzing/KLUE-TOD.git

2. pip install -r requirements.txt

3. cd KLUE-TOD

4. python interactive_all.py
```

## result
```
User: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요. 
Dialogue history: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요. 
dst : 관광-종류-랜드마크관광-지역-서울 북쪽
System :  동대문 (흥인지문) 어떠세요? 평점은 4점 입니다. 


User: 아 좋아요. 가격이 어떻게 되죠? 그리고 가까운 지하철역이 어딘가요?
Dialogue history: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요.  동대문 (흥인지문) 어떠세요? 평점은 4점 입니다. 아 좋아요. 가격이 어떻게 되죠? 그리고 가까운 지하철역이 어딘가요?
dst : 관광-이름-동대문 ( 평점 4점)
System :  입장료는 8000원 입니다. 가까운 지하철역은 동대문역으로 나오네요. 

User: 아 좋네요. 그러면 관광지랑 같은 지역에 있는 양식당중에 술도 팔고 야외석도 있는 곳이 있을까요?
Dialogue history: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요.  동대문 (흥인지문) 어떠세요? 평점은 4점 입니다. 아 좋아요. 가격이 어떻게 되죠? 그리고 가까운 지하철역이 어딘가요? 입장료는 8000원 입니다. 가까운 지하철역은 동대문역으로 나오네요. 아 좋네요. 그러면 관광지랑 같은 지역에 있는 양식당중에 술도 팔고 야외석도 있는 곳이 있을까요?
dst :  식당-주류 판매-yes식당-종류-양식당식당-지역-서울 북쪽식당-야외석 유무-yes
System :  가격대는 어떻게 할까요? 

User: 가격대는 자유롭게 알아봐주세요. 
Dialogue history: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요.  동대문 (흥인지문) 어떠세요? 평점은 4점 입니다. 아 좋아요. 가격이 어떻게 되죠? 그리고 가까운 지하철역이 어딘가요? 입장료는 8000원 입니다. 가까운 지하철역은 동대문역으로 나오네요. 아 좋네요. 그러면 관광지랑 같은 지역에 있는 양식당중에 술도 팔고 야외석도 있는 곳이 있을까요? 가격대는 어떻게 할까요? 가격대는 자유롭게 알아봐주세요. 
dst :  식당-가격대-dontcare식당-지역-서울 북쪽식당-종류-양식당식당-야외석 유무-yes식당-주류 판매-yes
System :  어차피자라는 한식당이 검색 되었습니다. 평점은 4 입니다

User: 음 찾아보니까 좋아보이네요. 금요일에 09시30분으로 8명 예약해주세요.
Dialogue history: 안녕하세요. 서울 북쪽에 있는 평이 좋은 관광지를 방문하고 싶어요. 랜드마크 타입으로 알아봐주세요.  동대문 (흥인지문) 어떠세요? 평점은 4점 입니다. 아 좋아요. 가격이 어떻게 되죠? 그리고 가까운 지하철역이 어딘가요? 입장료는 8000원 입니다. 가까운 지하철역은 동대문역으로 나오네요. 아 좋네요. 그러면 관광지랑 같은 지역에 있는 양식당중에 술도 팔고 야외석도 있는 곳이 있을까요? 가격대는 어떻게 할까요? 가격대는 자유롭게 알아봐주세요.  어차피자라는 한식당이 검색 되었습니다. 평점은 4 입니다. 음 찾아보니까 좋아보이네요. 금요일에 09시30분으로 8명 예약해주세요.
dst :  식당-예약 명수-8식당-예약 요일-금요일식당-이름-어차피자식당- 예약 시간-09:30
System :  예약이 되었습니다. 예약 번호는 HJOK4 입니다.

```


## Reference

[Huggingface](https://huggingface.co/)

[KLUE-wos](https://huggingface.co/datasets/klue/viewer/wos/train)

[KoGPT-2](https://huggingface.co/skt/kogpt2-base-v2)

[SimpleTOD paper](https://arxiv.org/pdf/2005.00796.pdf)
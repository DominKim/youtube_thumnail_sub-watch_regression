# youtube_thumnail_sub-watch_regression
  
- 이 프로젝트의 목적은 유튜브 썸네일 이미지를 활용하여 딥러닝(CNN) 회귀분석을 통해 종속변수(구독자수/조회수)를 예측하는 것이다.

- 또한, 미리 학습된 Resnet의 hidden layer에 수집한 이미지를 임베딩 하여 코사인 유사도를 통해 채널 마다 주 콘텐츠 혹은 자신이 만든 썸네일과 유명 유튜버들의 썸네일과의 유사도를 확인 할 수 있다.

## Dataset:
- Selenium을 활용하여 유튜브 카테고리, 썸네일, 조회수, 구독자수, 영상제목, 유튜버명을 수집하였다. 

- 파일 및 폴더 구조
``` python3
├── data
│   ├── youtube
|       ├── pkl
|       ├── thumnail
└── deeplearing_regression
|   ├── Jupyter_notebook_code
|       ├── 1_Crawler.ipynb
|       ├── Image_regression.ipynb
|       ├── Image_similarity.ipynb
|   ├── trainer.py
|   ├── train.py
|   ├── data_loader.py
|   ├── utils.py
|   ├── model_loader.py

```

## Pre-requisite

- Python 3.6 or higher
- PyTorch 1.6 or higher
- PyTorch Ignite
- PIL
- selenium

## Model:
![alt text](./figures/model_arch.png)

-  Our architecture can be seen in the figure above. It consists of several consecutive convolution layers. Another important point regarding the model is that instead of a classifier approach, we used a regression based model so that backpropagation flow starts from some continuous age value.

To see a more detailed tensorboard graph regarding our model, click [here.](./figures/tensorboard-graph.png)


## Results
- To decide on hyperparameters, we tried many different scenarios. Training and validation losses (Mean Average Error) for each scenario can be found [here.](./results/)

- The best results are taken when the hyperparameters are,  


| Hyperparameter| Choosen Value |
| -------------   | -------------      |
| Loss Function | Mean Sqaure Error	|
| Learning Rate | 0.0001   |
| Dropout Keep Probability | 0.6	|
| L2 Reg. Constant | 0.0001   |
| Batch Size | 200	|

- The corresponding results in our best model is given below, 

| Loss Type       | Mean Average Error |
| -------------   | -------------      |
| Validation Loss | 6.486	       |
| Test Loss 	  | 6.419	       |
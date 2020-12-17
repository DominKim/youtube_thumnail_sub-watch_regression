# youtube_thumnail_sub-watch_regression
  
- 이 프로젝트의 목적은 유튜브 썸네일 이미지를 활용하여 딥러닝(CNN) 회귀분석을 통해 종속변수(구독자수/조회수)를 예측하는 것이다.

- 또한, 미리 학습된 Resnet의 hidden layer에 수집한 이미지를 임베딩 하여 코사인 유사도를 통해 채널 마다 주 콘텐츠 혹은 자신이 만든 썸네일과 유명 유튜버들의 썸네일과의 유사도를 확인 할 수 있다.

## Dataset:
- Selenium을 활용하여 유튜브 카테고리, 썸네일, 조회수, 구독자수, 영상제목, 유튜버명을 수집하였다. 

- 파일 및 폴더 구조
``` python3
├── data
│   ├── youtube
│       ├── pkl
│       ├── thumnail
└── deeplearing_regression
│   ├── Jupyter_notebook_code
│       ├── 1_Crawler.ipynb
│       ├── Image_regression.ipynb
│       ├── Image_similarity.ipynb
│   ├── trainer.py
│   ├── train.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── model_loader.py
```

## Pre-requisite

- Python 3.6 or higher
- PyTorch 1.6 or higher
- PyTorch Ignite
- PIL
- selenium
```bash
(base) macui-MacBook-Pro-3:deeplearning_regression mac$ python ./train.py --model_fn resnet.pth --gpu_id -1 --n_epochs 10 --model_name resnet --n_classes 1 --freeze --use_pretrained
Namespace(batch_size=50, freeze=True, gpu_id=-1, model_fn='resnet.pth', model_name='resnet', n_classes=1, n_epochs=10, train_ratio=0.6, use_pretrained=True, valid_ratio=0.2, verbose=2)
Train: 1920
Valid: 640
Test: 640
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ~~~~
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1, bias=True)
)

```

## Model & Data:
### 1_Cralwer
- Selenium 패키지를 활용하여 youtube 썸네일 이미지를 저장하고, pickle 형식으로 이미지외 데이터를 저장

### data_loader
- resnet model을 수행하기 위해 이미지를 224, 244로 리사이즈를 하였고 y(종속변수)를 파이토치의 텐서 형식으로 바꾸고 파이토치 Dataloader를 이용해 train, valid, test set의로 나눔

### Image_similrity
- resnet 모델의 미리 학습된 avgpool layer를 가져와 이미지를 layer에 input하고 zero행렬에 embedding을 통해 이미지의 feature를 추출하고 코사인 유사도를 통해 가장 유사도가 높은 이미지를 추천

### model_loader
- 전이학습(transfer_learning)을 통해 회귀분석을 하기 위해 미리 학습된 resnet34(use_traine = True) 모델을 불러오고 학습된 파라미터들(weights)은 freeze 시키고 모델의 fc layer를 nn.Linear(n_features, config.n_classes)과 같이 변경하고 n_classes는 1로 설정(회귀분석)함.

## Why?
### Transfer Learing
- 전이학습을 활용한 이유는 여러 논문과 연구에서 나왔듯이 대량의 이미지를 미리 학습한 모델은 이미지의 특징(feature)을 추출하는 적절한 파라미터를 가지고 있기때문에 전이학습을 활용

- 또한, 위와같은 이유로 이미지 유사도를 추출하기 위해서 renset의 미리 학습된 layer를 활용

## Results 

| Hyperparameter| Choosen Value |
| -------------   | -------------      |
| Loss Function | Mean Sqaured Error	|
| Batch Size | 50	|
| n_Epoch | 10	|

- The corresponding results in our best model is given below, 

| Loss Type       | Mean squared Error |
| -------------   | -------------      |
| Validation Loss | 91.8	       |
| Train Loss 	  | 14.19	       |

- 컴퓨터 사양의 문제로 적은 데이터셋과 Epoch으로 학습을 수행

## Reference

- [Effortlessly Recommending Similar Images](https://towardsdatascience.com/effortlessly-recommending-similar-images-b65aff6aabfb)
- [TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
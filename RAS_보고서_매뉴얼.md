# 오픈소스 논문 분석 및 실행 매뉴얼

**대상 논문**: Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection  
**공식 코드**: Nothingtolose9979/RAS  
**과제 유형**: 오픈소스 논문 구현 및 분석  
**작성자**: [학번 / 이름 기입]

---

## 1. 과제 개요

본 보고서는 오픈소스 논문 **Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection**의 핵심 아이디어를 분석하고, 공개된 GitHub 코드를 실행하기 위한 매뉴얼을 정리한 것이다. 본 논문은 산업 이미지에서 결함을 탐지하는 **unsupervised anomaly detection** 문제를 다루며, 특히 여러 class를 하나의 모델로 처리하는 **unified unsupervised anomaly detection** setting에 초점을 둔다.

기존 anomaly detection 방법은 class마다 별도의 모델을 학습하는 **n-class-n-model** 방식이 많다. 하지만 실제 산업 현장에서는 검사 대상 종류가 많아질 수 있으므로, class마다 모델을 따로 관리하는 방식은 저장 공간, 관리 비용, 학습 비용 측면에서 비효율적이다. 이에 따라 하나의 모델로 여러 class를 동시에 처리하는 **n-class-one-model** 방식이 중요해지고 있다.

이 논문의 핵심은 feature reconstruction 과정에서 단순히 feature를 복원하는 것이 아니라, decoder layer들을 하나의 sequence처럼 보고 단계적으로 context를 보강하는 것이다. 이를 위해 논문은 **Reconstruction as Sequence (RAS)**라는 방법과 **RASFormer block**을 제안한다.

---

## 2. 논문 선택 이유

본 논문을 선택한 이유는 다음과 같다.

첫째, 과제 주제와 잘 맞는다. 공개 코드가 제공되어 있고, MVTec-AD 같은 산업 결함 이미지 dataset을 대상으로 실험할 수 있어 오픈소스 재현 과제에 적합하다.

둘째, unified anomaly detection이라는 현실적인 문제를 다룬다. 실제 제조 현장에서는 pill, cable, metal nut, screw처럼 다양한 제품을 검사해야 한다. class별로 모델을 따로 만드는 방식보다 하나의 모델로 여러 제품을 처리하는 방식이 더 실용적이다.

셋째, Transformer 기반 feature reconstruction을 사용하면서도 단순한 구조가 아니라, reconstruction process 자체를 sequence로 해석한다는 점이 흥미롭다. 기존 방식이 각 decoder layer를 독립적인 복원 단계처럼 사용하는 경향이 있었다면, RAS는 이전 단계에서 복원된 정보를 다음 단계에서 활용하도록 설계한다.

---

## 3. 연구 배경

### 3.1 Unsupervised Anomaly Detection

**Unsupervised anomaly detection**은 학습할 때 정상 sample만 사용하고, test 단계에서 정상과 비정상을 구분하는 task이다. 산업 결함 탐지에서는 모든 가능한 불량 유형을 미리 수집하기 어렵기 때문에 이 방식이 자주 사용된다.

기본 가정은 다음과 같다.

- 모델은 정상 sample만 보고 학습한다.
- 정상 영역은 잘 복원하거나 잘 표현할 수 있다.
- 학습 때 보지 못한 이상 영역은 복원 error 또는 feature discrepancy가 커진다.
- 이 차이를 이용해 anomaly score와 anomaly map을 만든다.

### 3.2 Unified Setting의 어려움

class별 모델에서는 하나의 class에 대한 정상 pattern만 학습하면 된다. 반면 unified setting에서는 여러 class의 정상 pattern을 동시에 학습해야 한다. 이때 정상 pattern의 다양성이 커지기 때문에 decoder가 너무 일반화되어 anomaly까지 잘 복원해버리는 문제가 생길 수 있다.

즉, anomaly detection에서는 모든 것을 잘 복원하는 모델이 좋은 모델이 아니다. 정상은 잘 복원하되, 이상 영역은 잘 복원하지 못해야 한다. 그러나 unified setting에서는 다양한 정상 sample을 학습하다 보니 decoder가 anomaly까지 복원하는 **identity mapping** 또는 과도한 일반화 문제가 발생할 수 있다.

### 3.3 Feature Reconstruction 기반 방법

논문은 image pixel 자체를 복원하는 방식보다, pre-trained backbone에서 추출한 feature를 복원하는 방식을 사용한다. 이 방식은 image의 low-level pixel보다 의미 있는 visual representation을 비교할 수 있기 때문에 anomaly detection에 유리하다.

RAS에서는 EfficientNet-B4 backbone으로 image feature를 추출하고, 여러 level의 feature를 resize 및 concatenate하여 original feature `f_org`를 만든다. 이후 encoder-decoder 구조를 통해 reconstructed feature `f_rec`를 생성하고, 두 feature의 차이를 anomaly score로 사용한다.

---

## 4. 제안 방법 분석

### 4.1 전체 구조

RAS의 전체 흐름은 다음과 같다.

1. 입력 image를 pre-trained CNN backbone에 넣어 feature를 추출한다.
2. 여러 level의 feature를 같은 크기로 맞춘 뒤 channel 방향으로 concatenate한다.
3. 학습 단계에서는 feature에 noise를 추가하여 denoising 방식으로 학습한다.
4. Transformer encoder가 noisy feature를 latent feature로 변환한다.
5. RASFormer decoder가 feature reconstruction을 sequence처럼 단계적으로 수행한다.
6. 최종 reconstructed feature와 original feature의 차이를 anomaly map으로 사용한다.

### 4.2 Reconstruction as Sequence

기존 reconstruction 방식은 decoder layer들을 단순히 여러 층으로 쌓아 feature를 복원한다. RAS는 이 과정을 sequence modeling 관점으로 다시 해석한다. 즉, 각 decoder layer를 하나의 time step으로 보고, 이전 step에서 얻은 latent feature를 다음 step에서 활용한다.

이 관점의 장점은 reconstruction 과정이 단순한 반복이 아니라, 이전 단계에서 이미 복원된 정보를 고려하면서 점진적으로 보정되는 과정이 된다는 점이다. 논문에서는 이를 통해 reconstruction quality가 향상되고, anomaly map도 더 정확해진다고 설명한다.

### 4.3 RASFormer Block

RASFormer block은 RAS의 핵심 구성 요소이다. 이 block은 두 가지 정보를 함께 사용한다.

- 이전 step의 latent feature `l`
- 현재 step의 contextual embedding `c`

여기서 contextual embedding은 현재 decoder step이 어떤 context를 복원해야 하는지에 대한 query 역할을 한다. 이전 latent feature는 지금까지 복원된 정보를 담고 있다.

### 4.4 Adaptive Gate

RASFormer block에서 중요한 부분은 **adaptive gate**이다. adaptive gate는 이전 latent feature 중에서 현재 step에 필요한 정보는 남기고, 불필요한 정보는 줄이는 역할을 한다.

수식으로는 다음과 같이 표현된다.

```text
a = sigmoid(W_A(l concat c))
l_A = a * l
```

쉽게 말하면, 이전 단계에서 가져온 정보를 그대로 전부 쓰는 것이 아니라 현재 context와 비교해서 필요한 부분만 통과시키는 구조이다. 이 덕분에 decoder가 이미 복원된 정보를 무의미하게 반복하지 않고, 다음 단계에서 더 필요한 정보를 중심으로 reconstruction할 수 있다.

### 4.5 Spatial Dynamics와 Sequential Dynamics

RASFormer는 두 가지 dynamics를 강화한다.

첫째, **sequential dynamics**이다. reconstruction이 여러 step으로 진행되므로, 이전 step에서 복원한 정보를 기억하고 다음 step에서 활용해야 한다. adaptive gate는 이전 정보를 조절하여 sequential dependency를 반영한다.

둘째, **spatial dynamics**이다. image feature는 위치별 정보를 가지고 있기 때문에, 각 region 간의 관계를 잘 파악해야 한다. RASFormer 내부의 multi-head self-attention은 서로 다른 image region 간의 관계를 학습할 수 있게 한다.

결국 RASFormer는 “이전 단계에서 무엇을 복원했는지”와 “현재 image region들이 서로 어떤 관계를 가지는지”를 함께 고려한다.

### 4.6 Loss와 Inference

학습에서는 original feature `f_org`와 reconstructed feature `f_rec` 사이의 MSE loss를 사용한다.

```text
L = ||f_org - f_rec||^2
```

Inference 단계에서는 두 feature의 차이를 L2 norm으로 계산하여 feature-level anomaly map을 만든다.

```text
S_feat = ||f_org - f_rec||_2
```

이후 anomaly map을 원본 image 크기로 upsampling하여 pixel-level anomaly map을 얻고, image-level anomaly score는 anomaly map에서 높은 값을 기반으로 계산한다.

---

## 5. 논문 실험 결과 요약

논문은 MVTec-AD, VisA, BTAD, MPDD dataset에서 RAS를 평가한다. 특히 MVTec-AD에서는 unified setting 기준으로 image-level AUROC 98.4, pixel-level AUROC 97.5를 보고한다. UniAD와 비교했을 때 image-level과 pixel-level 모두에서 성능 향상이 나타난다.

논문에서 중요한 실험 결과는 다음과 같이 정리할 수 있다.

| 항목 | 내용 |
|---|---|
| 주요 dataset | MVTec-AD, VisA, BTAD, MPDD |
| 주요 metric | Image-level AUROC, Pixel-level AUROC |
| MVTec-AD unified I-AUROC | 98.4 |
| MVTec-AD unified P-AUROC | 97.5 |
| 주요 비교 대상 | UniAD, DeSTSeg, SimpleNet 등 |
| 핵심 개선점 | feature reconstruction 과정의 context awareness 강화 |

Ablation study에서는 adaptive gate와 transformer를 모두 사용할 때 가장 좋은 성능을 보인다. 이는 RASFormer block에서 gate만 중요한 것이 아니라, attention을 통한 spatial relation modeling도 함께 필요하다는 것을 의미한다.

---

## 6. 오픈소스 코드 실행 매뉴얼

### 6.1 권장 실행 환경

공식 코드의 dependency는 최신 Python/torch 환경과 충돌할 수 있으므로, 가급적 conda 환경을 따로 만드는 것이 안전하다.

```bash
conda create -n ras python=3.8 -y
conda activate ras
```

PyTorch는 CUDA 환경에 맞추어 설치한다. 예시는 CUDA 11.6 기준이다.

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

그 다음 RAS repository를 clone하고 requirements를 설치한다.

```bash
git clone https://github.com/Nothingtolose9979/RAS.git
cd RAS
pip install -r requirements.txt
```

주의할 점은 Google Colab 기본 Python이 3.12인 경우가 많아 `torch==1.12.1` 설치가 실패할 수 있다는 점이다. 이 경우 local conda 환경이나 Python 3.8 기반 runtime을 사용하는 것이 더 안정적이다.

### 6.2 Dataset 준비

MVTec-AD dataset을 사용한다. 예를 들어 `pill` class를 사용할 경우 directory 구조는 다음과 같이 맞춘다.

```text
RAS/
└── data/
    └── mvtec/
        └── pill/
            ├── train/
            │   └── good/
            ├── test/
            │   ├── good/
            │   ├── crack/
            │   ├── color/
            │   └── ...
            └── ground_truth/
                ├── crack/
                ├── color/
                └── ...
```

`train/good`에는 정상 image만 들어간다. `test`에는 normal과 anomaly sample이 함께 들어가며, `ground_truth`에는 anomaly mask가 class별로 들어간다.

### 6.3 실행 예시

repository 버전에 따라 argument 이름이 다를 수 있으므로, 실제 실행 전에는 `train_val.py --help` 또는 README를 확인한다. 예시 command는 다음과 같다.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=10012 train_val.py \
  --config ./config/mvtec/config_mvtec.yaml \
  --clsname pill \
  --num_encoder_layers 2 \
  --num_decoder_layers 4 \
  --learning_rate 0.0007 \
  --seed 333
```

직접 `python train_val.py`로 실행할 경우 distributed 관련 환경변수 문제로 `RANK` 또는 `LOCAL_RANK` error가 발생할 수 있다. 이 경우 `torch.distributed.launch` 또는 `torchrun`을 사용하는 방식이 더 안정적이다.

### 6.4 결과 확인

실행이 정상적으로 완료되면 다음과 같은 결과를 확인한다.

- training log 출력
- image-level AUROC / pixel-level AUROC 출력
- anomaly map 또는 visualization image 저장
- `vis_compound/pill/...` 형태의 결과 image 생성 여부 확인

본 과제에서는 전체 dataset에 대한 완전한 성능 재현보다, open-source code를 실제로 실행하고 논문 구조와 결과 해석 방식을 이해하는 것이 중요하다. 따라서 실행 screenshot, terminal log, 결과 image를 보고서 또는 발표 자료에 함께 첨부하면 재현성을 설명하기 좋다.

### 6.5 자주 발생한 문제와 해결

| 문제 | 원인 | 해결 방법 |
|---|---|---|
| `No matching distribution found for torch==1.12.1` | Python version이 너무 높음 | Python 3.8 conda 환경 사용 |
| `KeyError: RANK` | distributed 실행 방식 문제 | `torch.distributed.launch` 또는 `torchrun` 사용 |
| dataset path error | config의 data path와 실제 폴더 구조 불일치 | `config_mvtec.yaml`와 `data/mvtec/pill` 구조 확인 |
| CUDA memory 부족 | batch size 또는 image size 문제 | batch size 감소, GPU runtime 확인 |
| metric이 논문과 다름 | seed, epoch, GPU, dataset split, preprocessing 차이 | 논문 수치와 직접 비교 시 조건 명시 |

---

## 7. 코드 분석 내용

### 7.1 Feature Extraction

RAS는 입력 image에서 backbone feature를 추출한다. feature는 여러 level에서 뽑히며, size를 맞춘 후 channel 방향으로 합쳐진다. 이렇게 하면 low-level edge/detail 정보와 high-level semantic 정보를 동시에 사용할 수 있다.

### 7.2 Noise 추가

학습 단계에서 feature에 noise를 추가하는 이유는 모델이 단순히 입력을 그대로 복사하지 않도록 하기 위해서이다. anomaly detection에서는 normal feature를 안정적으로 복원하는 능력이 필요하지만, 이상 feature까지 그대로 복원하면 anomaly score가 작아지는 문제가 생긴다. 따라서 noise를 넣고 이를 복원하게 하여 denoising 성격을 부여한다.

### 7.3 Encoder

Transformer encoder는 noise가 포함된 feature를 latent reconstruction space로 변환한다. 이때 attention을 통해 전체 feature token 사이의 관계를 반영한다.

### 7.4 Decoder와 RASFormer

Decoder는 RASFormer block으로 구성되어 있으며, 각 decoder layer는 sequence의 한 step처럼 동작한다. 이전 step의 latent feature와 현재 step의 contextual embedding을 함께 사용하여 feature를 복원한다.

### 7.5 Anomaly Map 생성

최종적으로 original feature와 reconstructed feature 사이의 차이를 계산한다. 정상 영역은 두 feature가 비슷하므로 score가 낮고, anomaly 영역은 reconstruction이 잘 되지 않아 score가 높게 나타난다. 이 score map을 image 크기로 키우면 pixel-level anomaly map이 된다.

---

## 8. 본인 구현 및 재현 정리 예시

본 과제에서는 RAS repository를 clone하고, MVTec-AD의 `pill` class를 중심으로 실행 환경을 구성하였다. 실행 과정에서 dependency 충돌과 distributed 실행 관련 error가 발생했으며, 이를 해결하기 위해 Python 3.8 기반 conda 환경과 PyTorch 1.12 계열 설치를 기준으로 정리하였다.

실행 결과는 terminal log와 visualization output을 통해 확인하였다. 특히 `vis_compound/pill/crack`과 같은 directory에 anomaly map이 저장되는 것을 확인할 수 있었고, 이를 통해 code가 정상적으로 실행되어 test image에 대한 anomaly localization 결과를 생성한다는 점을 확인하였다.

보고서에 첨부하면 좋은 자료는 다음과 같다.

- GitHub repository screenshot
- conda environment 또는 package version screenshot
- dataset directory tree screenshot
- 실행 command screenshot
- terminal log screenshot
- `pill` class anomaly map 결과 image
- AUROC 출력 결과 screenshot

---

## 9. 한계 및 개선 방향

첫째, 논문 수치를 완전히 재현하려면 충분한 GPU resource와 동일한 학습 조건이 필요하다. 본 과제에서는 제한된 환경에서 일부 class 중심으로 실행을 확인했기 때문에, 논문 전체 benchmark와 1:1로 비교하기에는 한계가 있다.

둘째, RAS는 reconstruction quality 향상에 초점을 둔 방법이므로, 정상과 비정상의 차이가 매우 미세하거나 background가 복잡한 경우에는 feature discrepancy가 충분히 크게 나타나지 않을 수 있다.

셋째, unified setting에서는 class 간 정상 pattern 차이가 매우 크기 때문에, 특정 class에서는 anomaly score threshold 설정이 어려울 수 있다. 실제 산업 현장에 적용하려면 class별 threshold calibration 또는 post-processing이 필요할 수 있다.

개선 방향으로는 backbone을 다른 foundation model로 바꾸어 비교하거나, RASFormer decoder layer 수를 조절하면서 성능과 inference time의 trade-off를 분석할 수 있다. 또한 MVTec-AD 전체 class를 모두 실행하여 논문 수치와 더 정량적으로 비교하는 실험도 가능하다.

---

## 10. 결론

RAS 논문은 unified unsupervised anomaly detection에서 feature reconstruction의 품질을 높이기 위해 reconstruction process를 sequence로 해석한 연구이다. 핵심은 RASFormer block과 adaptive gate를 통해 이전 reconstruction step의 정보를 조절하고, spatial relation과 sequential dependency를 함께 반영하는 것이다.

본 과제를 통해 단순히 논문을 읽는 것에서 끝나지 않고, 공개 code를 실행하기 위한 환경 구성, dataset 구조 확인, command 작성, 오류 해결 과정을 경험하였다. 이를 통해 open-source paper analysis에서 중요한 것은 최종 성능 수치뿐 아니라, 논문의 핵심 아이디어가 실제 code에서 어떤 구조로 구현되어 있는지 이해하는 것임을 확인하였다.

---

## 참고문헌

[1] Hui-Yue Yang et al., “Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection,” ECAI 2024 / arXiv:2409.06285.  
[2] Official GitHub Repository: Nothingtolose9979/RAS.  
[3] Paul Bergmann et al., “MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.  
[4] Zhiyuan You et al., “A Unified Model for Multi-class Anomaly Detection,” NeurIPS 2022.
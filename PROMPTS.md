# PROMPTS.md

## 오픈소스 논문 분석 과제 프롬프트 로그

- 대상 논문: **Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection**
- 공식 코드: **Nothingtolose9979/RAS**
- 사용 AI 도구: ChatGPT Codex
- 목적: 논문 이해, 오픈소스 코드 실행, 오류 해결, 결과 해석, 보고서 및 발표 자료 구성

> 아래 내용은 과제 수행 과정에서 사용한 프롬프트를 보고서 제출용으로 정리한 것이다. 실제 제출 전 본인이 사용한 날짜나 추가 질문이 있으면 수정해서 사용한다.

---

## 1. 논문 선정 및 과제 방향 설정

### Prompt 01
```text
오픈소스 논문 분석 과제로 anomaly detection 논문을 고르려고 해. 
Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection 논문이 과제에 적합한지 설명해줘.
```

**활용 결과**: 논문이 공개 코드가 있고, MVTec-AD 기반 실험이 가능하며, unified anomaly detection이라는 실제 산업적 문제를 다룬다는 점을 정리했다.

### Prompt 02
```text
이 논문을 발표 주제로 선택한 이유를 쉽게 설명해줘. 
너무 논문식 말고 학부생 발표에서 자연스럽게 말할 수 있게 써줘.
```

**활용 결과**: class별 모델을 따로 학습하는 방식의 한계와 unified model의 필요성을 발표 도입부로 구성했다.

---

## 2. 논문 내용 이해

### Prompt 03
```text
Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection 논문을 section별로 쉽게 설명해줘.
```

**활용 결과**: Introduction, Method, Experiment, Conclusion의 핵심 내용을 나누어 이해했다.

### Prompt 04
```text
RAS 논문에서 말하는 unified unsupervised anomaly detection이 뭔지 쉽게 설명해줘.
```

**활용 결과**: n-class-n-model과 n-class-one-model의 차이를 정리했다.

### Prompt 05
```text
feature reconstruction 기반 anomaly detection이 어떻게 anomaly map을 만드는지 쉽게 설명해줘.
```

**활용 결과**: original feature와 reconstructed feature의 차이를 이용해 anomaly score를 만든다는 흐름을 정리했다.

### Prompt 06
```text
RAS 논문에서 Reconstruction as Sequence가 정확히 무슨 뜻이야? 
decoder layer를 sequence로 본다는 말을 쉽게 설명해줘.
```

**활용 결과**: decoder layer를 시간 순서의 step처럼 보고, 이전 step 정보를 다음 step에서 활용한다는 식으로 이해했다.

### Prompt 07
```text
RASFormer block 구조를 adaptive gate, transformer, latent feature, contextual embedding 중심으로 설명해줘.
```

**활용 결과**: RASFormer block의 입력과 출력, adaptive gate의 역할을 정리했다.

### Prompt 08
```text
adaptive gate가 왜 필요한지 예시로 설명해줘. 
이전 reconstruction 정보를 그냥 쓰면 왜 안 되는지도 알려줘.
```

**활용 결과**: 이전 정보 중 필요한 부분만 통과시켜 reconstruction capacity를 효율적으로 쓰는 역할로 정리했다.

---

## 3. 코드 실행 환경 구성

### Prompt 09
```text
RAS GitHub 코드를 실행하려고 해. 
Python, PyTorch, CUDA 버전이 충돌하지 않게 conda 환경 구성 명령어를 정리해줘.
```

**활용 결과**: Python 3.8, PyTorch 1.12 계열 중심으로 실행 환경을 정리했다.

### Prompt 10
```text
Google Colab에서 torch==1.12.1 설치가 안 된다고 나와. 
왜 그런지와 해결 방법을 알려줘.
```

**활용 결과**: Colab 기본 Python version이 높아 old torch wheel과 맞지 않을 수 있음을 확인했고, local conda 환경 사용을 대안으로 정리했다.

### Prompt 11
```text
MVTec-AD pill dataset을 RAS 코드에서 사용할 때 폴더 구조를 어떻게 맞춰야 해?
```

**활용 결과**: `train/good`, `test/good`, `test/defect_type`, `ground_truth/defect_type` 구조를 정리했다.

---

## 4. 코드 실행 및 오류 해결

### Prompt 12
```text
RAS train_val.py를 실행했는데 KeyError: 'RANK'가 발생했어. 
이 오류가 왜 생기는지와 해결 방법을 알려줘.
```

**활용 결과**: distributed training 환경변수 문제로 이해했고, `torch.distributed.launch` 또는 `torchrun` 방식으로 실행해야 한다고 정리했다.

### Prompt 13
```text
RAS 코드를 MVTec-AD pill class 하나만 실행하는 command 예시를 만들어줘.
```

**활용 결과**: config file, clsname, encoder/decoder layer 수, learning rate, seed 등을 포함한 실행 command를 작성했다.

### Prompt 14
```text
실행 결과로 vis_compound/pill/crack 폴더에 이미지가 생성됐는데, 이걸 발표에서 어떻게 설명하면 돼?
```

**활용 결과**: code가 test image에 대해 anomaly map visualization을 생성한 것으로 설명했다.

### Prompt 15
```text
논문 수치랑 내 실행 결과가 다르면 보고서에 어떻게 써야 해?
```

**활용 결과**: seed, GPU, dataset split, epoch, dependency 차이 때문에 논문 수치와 완전히 같지 않을 수 있음을 한계로 정리했다.

---

## 5. 실험 결과 및 metric 해석

### Prompt 16
```text
AUROC가 anomaly detection에서 무슨 의미인지 한 줄로 쉽게 설명해줘.
```

**활용 결과**: 정상과 이상을 threshold 변화에 따라 얼마나 잘 구분하는지 나타내는 지표로 정리했다.

### Prompt 17
```text
image-level AUROC와 pixel-level AUROC 차이를 발표에서 쉽게 설명해줘.
```

**활용 결과**: image-level은 이미지 전체가 정상/비정상인지 판단하는 성능, pixel-level은 결함 위치를 얼마나 잘 찾는지 보는 성능으로 정리했다.

### Prompt 18
```text
RAS 논문의 ablation study에서 adaptive gate와 transformer가 각각 어떤 의미인지 설명해줘.
```

**활용 결과**: adaptive gate는 sequential dynamics, transformer는 spatial dynamics를 강화하는 요소로 정리했다.

---

## 6. AI 도구 활용에 대한 자기 평가

AI 도구는 논문 내용을 단순 요약하는 데만 사용하지 않고, 다음 목적에 활용하였다.

1. 논문 구조와 핵심 개념 이해
2. RASFormer, adaptive gate, reconstruction as sequence 개념 정리
3. GitHub 코드 실행 매뉴얼 작성
4. PyTorch, Python version, distributed 실행 오류 해결
5. AUROC 등 metric 해석
6. 보고서 목차 및 발표 흐름 구성

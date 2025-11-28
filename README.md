# 🧬 End-to-End AI Drug Discovery Pipeline for Rare Diseases
> **TTR 아밀로이드증(TTR Amyloidosis)** 치료를 위한 신규 약물 후보물질 발굴 및 최적화 파이프라인

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red) ![RDKit](https://img.shields.io/badge/Bio-RDKit-green)

## 📖 프로젝트 개요 (Overview)
희귀질환의 95%는 치료제가 없습니다. 본 프로젝트는 **생성형 AI(Generative AI)와 진화 알고리즘**을 결합하여, 기존 약물(Tafamidis)보다 효능은 높고 독성은 낮은 **TTR 아밀로이드증 치료제 후보물질**을 자동으로 탐색하는 End-to-End 파이프라인입니다.

### 🎯 핵심 성과 (Key Results)
* **High Efficacy:** 기존 약물 대비 결합 효율(LLE) 및 효능(Potency) 대폭 개선 (pKd 5.27 → 7.85)
* **Low Toxicity:** 심장 독성(hERG) 리스크 60% 감소 (0.67 → 0.25)
* **Novelty:** 기존 약물과 구조적 유사도 0.139의 **신규 골격(Novel Scaffold)** 발굴

---

## 🏗 파이프라인 아키텍처 (Pipeline Architecture)
이 파이프라인은 크게 **탐색(Discovery)**과 **최적화(Optimization)** 두 단계로 구성됩니다.

![Pipeline Architecture](./asset/flow.webp)


### 1. Data Preparation
* **Dataset Construction:** 단백질-리간드 복합체 데이터를 수집하고, 학습/검증 데이터셋의 구조적 다양성을 확보하기 위해 **Murcko Scaffold Split**을 수행했습니다.
* **Pocket Definition:**
    * PDB(Protein Data Bank)에서 TTR 결정 구조(Crystal Structure)를 확보했습니다.
    * **UCSF ChimeraX**를 활용하여 단백질 구조를 시각화하고, 공동 결정된(Co-crystallized) 리간드 좌표를 기반으로 결합 포켓(Binding Pocket)의 3D 좌표를 정밀하게 정의했습니다.

### 2. Generative Model Architecture (Pocket2Mol)
* **Architecture Overview:** 타겟 포켓의 기하학적 구조에 딱 맞는 분자를 생성하기 위해 **E(3)-Equivariant Graph Neural Network (GNN)** 기반의 **Pocket2Mol**을 활용했습니다.
* **Mechanism:**
    * **Encoder:** 포켓 원자들의 공간 좌표와 화학적 특징을 캡처하며, 회전 및 이동 불변성(Rotational and Translational Invariance, E(3)-symmetry)을 유지합니다.
    * **Decoder:** 포켓의 기하학적 임베딩을 조건(Condition)으로 받아, 높은 결합력을 가질 것으로 예측되는 리간드를 원자 단위(Atom-by-atom)로 생성합니다.

### 3. Screening & Optimization Strategy (The Loop)
생성된 분자들은 다단계 필터링 및 최적화 과정을 거칩니다:
1.  **Generation:** 생성 모델을 사용한 초기 분자 샘플링.
2.  **1st Filtering (Physicochemical):** ADMET 특성, QED(약물 유사성), SA Score(합성 가능성), RDKit Descriptors를 기반으로 1차 필터링 수행.
3.  **2nd Filtering (Affinity):** 자체 개발한 GNN 기반 예측 모델을 통해 결합력 예측.
4.  **Validation:** 딥러닝 기반 도킹 툴인 **Gnina**를 사용하여 결합 모드 및 스코어 최종 검증.
5.  **Evolutionary Optimization:**
    * 선별된 고득점 후보물질을 **유전 알고리즘(Evolutionary Algorithm)**에 투입.
    * 시스템이 반복적으로 분자를 변형(Mutation) 및 최적화하여 적합도 점수(Fitness Score = 효능 + 안전성)를 극대화하는 폐루프(Closed-loop) 구조.

### 4. Scoring & Validation Models (Critic)
생성-최적화 루프 내에서 "Critic" 역할을 수행하는 커스텀 GNN 모델을 개발했습니다:
* **Binding Affinity Model:** pKd 값을 예측하여 약물의 효능(Potency) 추정.
* **ADMET Model:** hERG 저해와 같은 독성 리스크 및 약동학적 프로필 예측.

---

## 🧪 핵심 기술 (Technical Highlights)

### 1. Hybrid GNN for ADMET
* **문제:** 기존 GNN은 분자 구조만 학습하여 구체적인 물리화학적 특성을 놓칠 수 있음
* **해결:** 구조적 임베딩(GNN)에 **15가지 물리화학적 특징(Descriptors)**을 결합(Concatenation)
* **결과:** hERG 독성 예측 **ROC-AUC 0.789** 달성

### 2. Scaffold Split Validation
* **문제:** Random Split은 구조가 비슷한 분자에 과적합(Overfitting) 발생 가능
* **해결:** **Murcko Scaffold Split**을 적용하여 학습/검증 데이터의 골격을 분리
* **결과:** "보지 못한 구조(Novel Structure)"에 대한 일반화 성능(Generalization) 확보

### 3. Engineering Optimization
* **Ensemble:** 3-Seed Ensemble로 예측 편향(Bias) 최소화
* **Gradient Accumulation:** 제한된 GPU 메모리에서 배치 사이즈 효율화

---

## 실험 환경 (Experimental Setup)
To enable efficient training and inference of 3D-GNN models, I built a custom deep learning environment from scratch.

* **Hardware (Custom Workstation):**
    * **GPU:** NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
    * **CPU:** AMD Ryzen 7 7700 8-Core Processor
    * **RAM:** 32GB
* **Software Environment:**
    * **OS:** Linux (Ubuntu 22.04 LTS) - configured for optimal GPU utilization.
    * **CUDA Version:** 12.9 (Driver Support)
* **Engineering Highlights:**
    * **Dependency Management:** Replaced Conda with **Mamba** to resolve complex dependency conflicts between `rdkit`, `pytorch-geometric`, and `biopython`, reducing environment setup time by **90%**.
    * **Troubleshooting:** Resolved `CUDA device-side assert` errors caused by graph dimension mismatches during the training of GNN layers (Documented in `Debugging_log.ipynb`).

## 📊 실험 결과 (Results)

최종 도출된 후보물질(MOL_1282)은 대조군(Tafamidis) 대비 압도적인 성능을 보였습니다.

| 구분 | 대조군 (Tafamidis) | **최종 후보 (MOL_1282)** | 비고 |
| :--- | :---: | :---: | :--- |
| **Potency (pKd)** | 5.269 | **7.847** | 100배 이상 결합 상수 예측 |
| **Efficiency (LLE)** | 3.47 | **5.49** | 결합 효율 개선 |
| **Toxicity (hERG)** | 0.67 (High Risk) | **0.25 (Low Risk)** | 독성 60% 감소 |
| **Novelty** | 1.0 | **0.139** | 특허 회피 가능 (신규 골격) |


![Docking Result](./asset/docking.webp)

*Figure 1. 3D Docking Simulation of MOL_1282. The candidate forms a strong interaction with the hydrophobic cavity, resulting in 100x higher potency than Tafamidis.*

### 📋 전체 후보물질 리스트 (Full Candidates Data)
최종 선별된 34개 후보물질의 상세 물성 데이터(SMILES, pKd, LLE, hERG 등)는 아래 링크에서 확인할 수 있습니다.

👉 **[전체 34개 후보물질 리스트 확인하기](./results/final_candidates.csv)**

---

## 🛠 기술 스택 (Tech Stack)
* **Languages:** Python
* **Environment:** Anaconda (Conda), Python 3.8
* **AI/ML:** PyTorch, PyTorch Geometric, Optuna
* **Cheminformatics:** RDKit, OpenBabel
* **Bioinformatics:** PyMOL, ChimeraX, Gnina, Pocket2Mol
* **Tools:** Jupyter Lab, Datawarrior

---

## 📂 프로젝트 구조 (Directory Structure)

```text
├── 01_Pipeline/                 # 메인 실행 파이프라인
│   ├── B_Run_Pipeline.ipynb     # Phase 1: 초기 탐색 및 필터링
│   ├── G_DrugPipelineCycle.ipynb # Phase 2: 진화적 최적화 사이클
│   ├── config/                  # 설정 파일
│   └── utils/                   # 유틸리티 함수
├── 02_Models/                   # 모델 학습 및 평가 연구용 (Research)
│   ├── ADMET/                   # 독성/물성 예측 모델
│   ├── Binding_Affinity/        # 결합력 예측 모델
│   └── Pocket2Mol/              # 분자 생성 모델
├── models/                      # 학습 완료된 모델 가중치 (.pt 파일)
├── data/                        # 입력 데이터 (PDB 등)
└── requirements.txt             # 의존성 패키지
```

---

## 🚀 설치 및 실행 (Installation & Usage)

본 프로젝트는 **Generative Model(Pocket2Mol)**과 **Predictive Model(DrugAI)**의 라이브러리 버전 호환성을 위해 **두 개의 독립된 Conda 가상환경**을 사용합니다.

### 1. 환경 설정 (Environment Setup)

먼저 리포지토리를 복제하고, 제공된 `.yml` 파일을 이용해 두 개의 가상환경을 생성합니다.

```bash
# 1. Repository 복제
git clone [https://github.com/username/rare-disease-drug-discovery.git](https://github.com/username/rare-disease-drug-discovery.git)
cd rare-disease-drug-discovery

# 2. 'Pocket2Mol' 환경 설치 (분자 생성용)
conda env create -f environment_pocket2mol.yml

# 3. 'drugai' 환경 설치 (물성 예측 및 최적화용)
conda env create -f environment_drugai.yml
```

---

### 2. 파이프라인 실행 (Run Pipeline)

이 파이프라인은 Jupyter Notebook으로 실행됩니다. 단계에 맞춰 **알맞은 가상환경을 활성화(Activate)**한 후 노트북을 실행해 주세요.

#### **Step 1: 후보물질 생성 및 1차 필터링 (Phase 1)**
* **사용 환경:** `Pocket2Mol` (또는 `drugai` - *본인이 사용하는 환경에 맞춰 선택*)
* **실행 파일:** `01_Pipeline/B_Run_Pipeline.ipynb`
* **실행 방법:**
  ```bash
  conda activate Pocket2Mol
  jupyter lab
  # 위 노트북을 열어서 실행
  ```
* **기능:** TTR 타겟 구조를 기반으로 새로운 분자를 생성하고, 기본적인 물리화학적 필터링을 수행합니다.

#### **Step 2: 진화적 최적화 및 최종 검증 (Phase 2)**
* **사용 환경:** `drugai`
* **실행 파일:** `01_Pipeline/G_DrugPipelineCycle.ipynb`
* **실행 방법:**
  ```bash
  conda activate drugai
  jupyter lab
  # 위 노트북을 열어서 실행
  ```
* **기능:** Step 1에서 선별된 물질을 시드(Seed)로 하여, 유전 알고리즘(Genetic Algorithm)을 통해 효능과 안전성을 최적화합니다.

---

### 3. 개별 모델 학습 (Optional)
모델 연구 및 재학습을 위해서는 `drugai` 환경을 사용합니다.
* **ADMET 모델:** `02_Models/ADMET/notebooks/ADMET_model_eval.ipynb`
* **결합력(Affinity) 모델:** `02_Models/Binding_Affinity/notebooks/binding affinity(ML).ipynb`

---

# GCSFT

**G**eneralizable **C**onfidence **S**upervised **F**ine-**T**uning

---

## 🚀 Installation

```bash
conda create -n gcsft python=3.10
conda activate gcsft
pip install -r requirements.txt
```

---

## 📊 Baselines

### Model Types
- **Pretrained model** - Base evaluation
- **Probing** - Linear probing approach
- **SFT baseline** - Supervised fine-tuning
  - [Teaching Models to Express Their Uncertainty in Words](https://github.com/sylinrl/CalibratedMath)
- **RL baseline** - Reinforcement learning approach
  - [RLCR](https://github.com/damanimehul/RLCR)
  - [Reward Doubt](https://arxiv.org/pdf/2503.02623)

---

## 🤖 Models

| Model | Size |
|-------|------|
| LLaMA-3.2-3B-Instruct | Small size |
| Qwen-3-8B | Medicum size | 
| GPT-oss-20B | Large size | 

---

## 📚 Datasets

### Cluster 1: Internal Knowledge based Problem
- **Training**: GSM8K, MATH, BigMath
- **Evaluation**: TBD

### Cluster 2: External Evidence based Problem 
- **Training**: NIAH
- **Evaluation**: TBD

### Cluster 3: Mixed Problem
- **Training**: -
- **Evaluation**: ContractNLI, HotpotQA
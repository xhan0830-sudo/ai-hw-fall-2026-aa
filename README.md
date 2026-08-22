# Adversarial Attacks on MNIST (`ai-hw-fall-2026-aa`)

## Clean Model Performance
- **Clean Recognition Rate:** 98.85%

## Experimental Results: Attack Success Rate (ASR)

| Attack Method | Epsilon ($\epsilon = 0.1$) | Epsilon ($\epsilon = 0.2$) | Epsilon ($\epsilon = 0.3$) |
| :--- | :---: | :---: | :---: |
| **FGSM** | 18.2% | 58.4% | 85.1% |
| **I-FGSM / PGD** | 42.6% | 94.8% | 99.8% |
| **Momentum I-FGSM** | 46.1% | 96.5% | 99.9% |

## How to Run

1. Install dependencies:
   `pip install -r requirements.txt`
2. Train model:
   `python train.py`
3. Run attack benchmarks:
   `python evaluate.py`

# Module 3: Model Evasion Attacks

## Overview

This module covers adversarial evasion attacks against machine learning models, including both computer vision and natural language processing systems. You'll learn white-box and black-box attack techniques, understand transferability, and explore certified defenses.

## Quick Start

**Recommended Learning Path**:
1. Start with [Lab 1: White-Box Evasion](labs/lab1_whitebox_evasion.ipynb) - Learn FGSM and PGD attacks
2. Read [Adversarial NLP](adversarial_nlp.ipynb) - Understand text-based attacks
3. Complete [Lab 3: Text Attacks](labs/lab3_text_attacks.ipynb) - Practice NLP attacks
4. Read [Certified Defenses](certified_defenses.ipynb) - Learn provable robustness
5. Complete remaining labs and check [ANSWERS.ipynb](labs/ANSWERS.ipynb) for solutions

## Learning Objectives

By the end of this module, you will be able to:
- Execute white-box attacks (FGSM, PGD, C&W)
- Perform black-box attacks (HopSkipJump, query-based)
- Craft adversarial examples for NLP systems
- Understand attack transferability
- Implement certified defenses
- Evaluate model robustness

## Module Contents

### Interactive Theory Notebooks

These notebooks contain comprehensive theory with runnable code examples. Work through these to understand the concepts before attempting the labs.

**📖 New to these notebooks?** See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for tips on how to use them effectively.

1. **[Adversarial NLP](adversarial_nlp.ipynb)**
   - Character, word, and sentence-level attacks
   - Gradient-based NLP attacks (HotFlip, TextFooler, BERT-Attack)
   - Defense mechanisms for text models
   - 30 runnable code examples

2. **[Certified Defenses](certified_defenses.ipynb)**
   - **"Certified" = Mathematical proof of robustness** (not just tested, but proven!)
   - Randomized smoothing
   - Interval bound propagation
   - Lipschitz constraints
   - Provable robustness guarantees against ALL attacks
   - 23 runnable code examples

### Hands-On Labs

Complete these labs to practice implementing attacks and defenses:

1. [Lab 1: White-Box Evasion](labs/lab1_whitebox_evasion.ipynb)
2. [Lab 2: Black-Box Evasion](labs/lab2_blackbox_evasion.ipynb)
3. [Lab 3: Text Adversarial Attacks](labs/lab3_text_attacks.ipynb)
4. [Lab 4: Transfer Attacks](labs/lab4_transfer_attacks.ipynb)
5. [Lab 5: Certified Robustness](labs/lab5_certified_robustness.ipynb)

### Answer Guide

- **[ANSWERS.ipynb](labs/ANSWERS.ipynb)** - Complete solutions for all lab exercises

### Implementation Notes
- HopSkipJump and other advanced attacks are implemented directly in the lab notebooks
- All utility functions are contained within the individual lab files

## Topics Covered

### 1. White-Box Attacks (Computer Vision)

**Gradient-Based Attacks**:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Carlini & Wagner (C&W)
- DeepFool
- Momentum Iterative Method (MIM)

**Optimization-Based Attacks**:
- L2, L∞, L0 attacks
- Targeted vs untargeted
- Adversarial patch attacks

### 2. Black-Box Attacks

**Query-Based Attacks**:
- HopSkipJump
- Boundary Attack
- Square Attack
- SimBA

**Transfer-Based Attacks**:
- Model ensemble attacks
- Cross-architecture transfer
- Boosting transferability

**Score-Based vs Decision-Based**:
- Gradient estimation
- Query efficiency
- Practical constraints

### 3. Adversarial NLP

**Character-Level Attacks**:
- Homoglyph substitution
- Character insertion/deletion
- Keyboard typos

**Word-Level Attacks**:
- Synonym substitution
- Embedding-based substitution
- Word insertion/deletion

**Sentence-Level Attacks**:
- Paraphrasing
- Back-translation
- Style transfer

**Gradient-Based NLP**:
- HotFlip
- TextFooler
- BERT-Attack

### 4. Certified Defenses

**Randomized Smoothing**:
- Gaussian noise smoothing
- Certified L2 radius
- Training for smoothing

**Interval Bound Propagation**:
- Forward bound propagation
- Backward bound propagation
- CROWN certification

**Lipschitz Constraints**:
- Spectral normalization
- Lipschitz-constrained layers
- Certified radius computation

**Convex Relaxations**:
- Linear relaxation of ReLU
- Convex outer approximation
- Verification methods

## Attack Taxonomy

```
Evasion Attacks
├── White-Box (Full Model Access)
│   ├── Gradient-Based
│   │   ├── FGSM
│   │   ├── PGD
│   │   ├── C&W
│   │   └── MIM
│   └── Optimization-Based
│       ├── L2 Attack
│       ├── L∞ Attack
│       └── L0 Attack
├── Black-Box (No Model Access)
│   ├── Transfer-Based
│   │   ├── Ensemble Transfer
│   │   └── Cross-Architecture
│   ├── Query-Based
│   │   ├── Score-Based (HopSkipJump)
│   │   ├── Decision-Based (Boundary)
│   │   └── Gradient-Free (SimBA)
│   └── Hybrid Approaches
├── NLP-Specific
│   ├── Character-Level
│   ├── Word-Level
│   ├── Sentence-Level
│   └── Gradient-Based (HotFlip, TextFooler)
└── Certified Defenses
    ├── Randomized Smoothing
    ├── Interval Bound Propagation
    ├── Lipschitz Constraints
    └── Convex Relaxations
```

## Prerequisites

- Completion of Module 1 (Introduction)
- Understanding of gradient descent
- Basic linear algebra
- Python and PyTorch/TensorFlow
- For NLP: Understanding of transformers

## Estimated Time

- Theory: 8-10 hours
- Labs: 10-12 hours
- Total: 18-22 hours

## Learning Path

### Beginner Track
1. Start with Lab 1 (White-Box Attacks)
2. Read White-Box theory
3. Understand FGSM and PGD
4. Practice on MNIST/CIFAR-10

### Intermediate Track
1. Complete Beginner Track
2. Study Black-Box attacks
3. Complete Lab 2 (Black-Box)
4. Explore transfer attacks

### Advanced Track
1. Complete Intermediate Track
2. Study Adversarial NLP
3. Complete Lab 3 (Text Attacks)
4. Study Certified Defenses
5. Complete Lab 5 (Certification)

## Practical Applications

### Security Testing
- Test model robustness
- Identify vulnerabilities
- Evaluate defenses
- Red team exercises

### Real-World Scenarios
- Autonomous vehicles (stop sign attacks)
- Face recognition (adversarial faces)
- Spam filters (evasion)
- Malware detection (adversarial malware)
- Content moderation (toxic text evasion)

## Tools and Frameworks

### Attack Libraries
```python
# Adversarial Robustness Toolbox (ART)
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# Foolbox
import foolbox as fb
attack = fb.attacks.LinfPGD()

# CleverHans
from cleverhans.torch.attacks import fast_gradient_method

# TextAttack (for NLP)
from textattack.attack_recipes import TextFoolerJin2019
```

### Defense Libraries
```python
# Randomized Smoothing
from smoothing import Smooth

# AutoLiRPA (Certification)
from auto_LiRPA import BoundedModule

# Adversarial Training
from robustness import train_model
```

## Assessment

### Knowledge Check
- Explain difference between white-box and black-box attacks
- Describe FGSM algorithm
- Compare PGD and C&W attacks
- Explain attack transferability
- Describe certified defenses

### Practical Assessment
1. Generate adversarial examples using FGSM
2. Implement PGD attack from scratch
3. Execute black-box attack with limited queries
4. Craft adversarial text examples
5. Evaluate certified robustness
6. Compare attack success rates

### Success Criteria
- Successfully fool model with adversarial examples
- Achieve >80% attack success rate
- Generate imperceptible perturbations
- Understand defense mechanisms
- Implement basic certified defense

## Common Pitfalls

1. **Too Large Perturbations**
   - Solution: Use appropriate epsilon values
   - Validate imperceptibility

2. **Ignoring Defenses**
   - Solution: Test against defended models
   - Understand adaptive attacks

3. **Overfitting to Specific Model**
   - Solution: Test transferability
   - Use ensemble attacks

4. **Neglecting Semantic Constraints (NLP)**
   - Solution: Preserve meaning
   - Check grammaticality

5. **Unrealistic Threat Models**
   - Solution: Consider practical constraints
   - Align with real-world scenarios

## Advanced Topics

### Adaptive Attacks
- Attacking defended models
- Gradient obfuscation
- Expectation over Transformation (EOT)

### Physical Adversarial Examples
- Adversarial patches
- 3D adversarial objects
- Robust physical perturbations

### Universal Adversarial Perturbations
- Single perturbation for multiple inputs
- Generalization across models

### Adversarial Training
- Training on adversarial examples
- PGD adversarial training
- TRADES algorithm

## Real-World Case Studies

### Case 1: Stop Sign Attack (2018)
**Attack**: Adversarial stickers on stop sign
**Impact**: Autonomous vehicle misclassification
**Defense**: Robust training, physical defenses
**Source**: Eykholt et al. "Robust Physical-World Attacks on Deep Learning Visual Classification"

### Case 2: Face Recognition Evasion (2016)
**Attack**: Adversarial glasses
**Impact**: Impersonation attacks
**Defense**: Multi-view verification
**Source**: Sharif et al. "Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition"

### Case 3: Spam Filter Evasion (Ongoing)
**Attack**: Character substitution, paraphrasing
**Impact**: Spam bypasses filters
**Defense**: Preprocessing, adversarial training
**Source**: Various industry reports and academic studies on email security

### Case 4: Malware Detection Evasion (2019)
**Attack**: Adversarial malware variants
**Impact**: Evades ML-based detection
**Defense**: Ensemble methods, feature engineering
**Source**: Pierazzi et al. "Intriguing Properties of Adversarial ML Attacks in the Problem Space"

## Research Directions

### Open Problems
1. Scalable certified defenses for large models
2. Certified robustness for NLP
3. Physical adversarial examples
4. Adversarial examples in production
5. Theoretical understanding of adversarial examples

### Emerging Areas
- Adversarial examples for LLMs
- Multimodal adversarial attacks
- Adversarial examples in RL
- Quantum adversarial examples

## Additional Resources

### Papers
- "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015) - FGSM
- "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
- "Certified Adversarial Robustness via Randomized Smoothing" (Cohen et al., 2019)
- "Is BERT Really Robust?" (Jin et al., 2020) - TextFooler

### Books
- "Adversarial Robustness for Machine Learning" (Biggio & Roli, 2018)
- "Adversarial Machine Learning" (Joseph et al., 2019)

### Online Resources
- [Adversarial ML Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)
- [RobustML](https://www.robust-ml.org/)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## Next Steps

After completing this module:
1. Understand all major evasion attack types
2. Implement attacks from scratch
3. Evaluate model robustness
4. Explore certified defenses
5. Proceed to [Module 4: Data Extraction & Privacy](../04_data_extraction/README.md)

---

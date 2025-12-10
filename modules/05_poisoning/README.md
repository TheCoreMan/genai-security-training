# Module 5: Model Poisoning & Backdoor Attacks

## Overview

This module covers advanced poisoning attacks that compromise models during training. These are among the most dangerous attacks as they can persist undetected in production systems and be triggered at will by adversaries.

## Learning Objectives

By the end of this module, you will be able to:
- Execute data poisoning attacks on training datasets
- Implement backdoor attacks with various trigger mechanisms
- Understand supply chain vulnerabilities in ML pipelines
- Detect poisoned models and backdoors
- Design robust defenses against poisoning attacks
- Assess supply chain security risks

## Module Contents

### Theory
Comprehensive theory is integrated throughout this README, covering:
- Data poisoning fundamentals and advanced techniques
- Backdoor attack methods and trigger design
- Supply chain vulnerabilities and attack vectors
- Detection techniques and mitigation strategies

### Labs
1. [Lab 1: Data Poisoning](labs/lab1_data_poisoning.ipynb)
2. [Lab 2: Backdoor Attacks](labs/lab2_backdoor_attacks.ipynb)
3. [Lab 3: LLM Poisoning](labs/lab3_llm_poisoning.ipynb)
4. [Lab 4: Defense & Detection](labs/lab4_defense_detection.ipynb)

## Expert-Level Concepts

### Advanced Poisoning Techniques

#### 1. Clean-Label Poisoning
**Difficulty**: Expert
**Concept**: Poison training data without changing labels
```python
# Adversary adds imperceptible perturbations
# Labels remain correct
# Model learns to associate trigger with target class
```

**Why It's Dangerous**:
- Bypasses label verification
- Difficult to detect visually
- Survives data sanitization
- Effective with small poison rates

#### 2. Gradient-Based Poisoning
**Difficulty**: Expert
**Concept**: Optimize poison samples using gradients
```python
# Maximize impact on model parameters
# Minimize detection probability
# Target specific model behaviors
```

#### 3. Federated Learning Poisoning
**Difficulty**: Expert
**Concept**: Poison model updates in distributed training
```python
# Compromise local models
# Aggregate malicious updates
# Evade Byzantine-robust aggregation
```

### Advanced Backdoor Techniques

#### 1. Semantic Backdoors
**Difficulty**: Expert
**Concept**: Use natural features as triggers
```python
# Trigger: "Wearing sunglasses"
# Target: Misclassify as specific person
# Advantage: Naturally occurring, hard to detect
```

#### 2. Dynamic Backdoors
**Difficulty**: Expert
**Concept**: Triggers that change over time
```python
# Adaptive to defenses
# Context-dependent activation
# Multiple trigger variants
```

#### 3. Composite Backdoors
**Difficulty**: Expert
**Concept**: Multiple conditions required for activation
```python
# Trigger: Specific word + specific context + specific time
# Harder to detect through random testing
# More stealthy activation
```

### Supply Chain Attack Vectors

#### 1. Model Repository Poisoning
**Target**: Pre-trained models (HuggingFace, etc.)
**Method**: Upload poisoned models
**Impact**: Widespread compromise

#### 2. Dataset Poisoning
**Target**: Public datasets
**Method**: Contribute poisoned samples
**Impact**: Affects all downstream users

#### 3. Library Compromise
**Target**: ML frameworks and dependencies
**Method**: Malicious code injection
**Impact**: Arbitrary code execution

#### 4. Serialization Exploits
**Target**: Model loading mechanisms
**Method**: Exploit pickle, unsafe deserialization
**Impact**: Remote code execution

## Real-World Case Studies

### Case 1: BadNets (2017)
**Attack**: Backdoor in traffic sign recognition
**Trigger**: Small sticker on stop sign
**Result**: Misclassified as speed limit
**Impact**: Demonstrated real-world danger
**Source**: Gu et al. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"

### Case 2: TrojanNet (2020)
**Attack**: Backdoor in facial recognition
**Trigger**: Specific glasses pattern
**Result**: Bypass authentication
**Impact**: Security system compromise
**Source**: Liu et al. "Trojaning Attack on Neural Networks"

### Case 3: Supply Chain Attack on PyTorch (2023)
**Attack**: Malicious dependency in torchtriton
**Trigger**: Model loading
**Result**: Data exfiltration
**Impact**: Widespread exposure
**Source**: PyTorch Security Advisory and industry reports

### Case 4: Federated Learning Poisoning (2021)
**Attack**: Model poisoning in healthcare FL
**Trigger**: Specific patient demographics
**Result**: Biased diagnoses
**Impact**: Patient safety concerns
**Source**: Bagdasaryan et al. "How To Backdoor Federated Learning"

## Attack Taxonomy

```
Poisoning Attacks
├── Data Poisoning
│   ├── Label Flipping
│   ├── Clean-Label Poisoning
│   ├── Gradient-Based Poisoning
│   └── Feature Poisoning
├── Backdoor Attacks
│   ├── Patch-Based Backdoors
│   ├── Semantic Backdoors
│   ├── Dynamic Backdoors
│   ├── Composite Backdoors
│   └── Physical Backdoors
├── Supply Chain Attacks
│   ├── Model Repository Poisoning
│   ├── Dataset Poisoning
│   ├── Library Compromise
│   └── Serialization Exploits
└── Federated Learning Attacks
    ├── Model Update Poisoning
    ├── Byzantine Attacks
    └── Sybil Attacks
```

## Detection Techniques

### 1. Activation Clustering
**Method**: Cluster activations to find outliers
**Effectiveness**: Good for patch-based backdoors
**Limitation**: Fails on semantic backdoors

### 2. Neural Cleanse
**Method**: Reverse-engineer potential triggers
**Effectiveness**: Detects various backdoor types
**Limitation**: Computationally expensive

### 3. STRIP (STRong Intentional Perturbation)
**Method**: Test input sensitivity to perturbations
**Effectiveness**: Runtime detection
**Limitation**: Can be evaded

### 4. Spectral Signatures
**Method**: Analyze representation space
**Effectiveness**: Detects poisoned samples
**Limitation**: Requires clean reference data

### 5. Fine-Pruning
**Method**: Prune neurons and fine-tune
**Effectiveness**: Removes backdoors
**Limitation**: May reduce accuracy

## Defense Strategies

### Training-Time Defenses

#### 1. Data Sanitization
```python
# Remove outliers
# Verify labels
# Check for duplicates
# Validate sources
```

#### 2. Robust Training
```python
# Differential privacy
# Certified training
# Adversarial training
# Ensemble methods
```

#### 3. Anomaly Detection
```python
# Monitor training dynamics
# Detect unusual gradients
# Flag suspicious samples
# Validate model updates
```

### Inference-Time Defenses

#### 1. Input Preprocessing
```python
# Remove potential triggers
# Normalize inputs
# Apply transformations
# Detect anomalies
```

#### 2. Model Monitoring
```python
# Track prediction patterns
# Detect unusual activations
# Monitor confidence scores
# Alert on anomalies
```

#### 3. Ensemble Voting
```python
# Use multiple models
# Majority voting
# Detect disagreements
# Reduce single-point failure
```

### Supply Chain Defenses

#### 1. Model Verification
```python
# Scan for backdoors
# Verify checksums
# Test on validation sets
# Monitor behavior
```

#### 2. Secure Serialization
```python
# Use SafeTensors
# Avoid pickle
# Validate model files
# Sandbox loading
```

#### 3. Dependency Management
```python
# Pin versions
# Verify signatures
# Use private registries
# Regular audits
```

## Expert-Level Labs

### Lab 1: Data Poisoning
**Difficulty**: Advanced
- Implement gradient-based poisoning
- Execute clean-label attacks
- Test against defenses
- Measure attack success rate

### Lab 2: Backdoor Attacks
**Difficulty**: Advanced
- Design effective triggers
- Inject backdoors during training
- Test activation reliability
- Evade detection mechanisms

### Lab 3: LLM Poisoning
**Difficulty**: Advanced
- Poison language model training
- Implement instruction backdoors
- Test trigger activation
- Analyze attack persistence

### Lab 4: Defense & Detection
**Difficulty**: Advanced
- Implement Neural Cleanse
- Test activation clustering
- Develop custom detection
- Evaluate defense effectiveness

## Prerequisites

- Completion of Modules 1-4
- Strong understanding of deep learning
- Experience with PyTorch/TensorFlow
- Knowledge of optimization techniques
- Understanding of adversarial ML

## Estimated Time

- Theory: 5-6 hours
- Labs: 10-12 hours
- Total: 15-18 hours

## Assessment

### Knowledge Check
- Poisoning attack types
- Backdoor mechanisms
- Detection techniques
- Defense strategies

### Practical Assessment
1. Execute data poisoning attack
2. Implement backdoor attacks with custom triggers
3. Demonstrate LLM poisoning techniques
4. Detect backdoors in provided models
5. Design comprehensive defense strategy

## Success Criteria

- Successfully poison training data (>80% attack success)
- Implement backdoor with <5% accuracy drop
- Detect backdoors with >90% accuracy
- Design defense that reduces attack success to <20%
- Complete supply chain security assessment

## Advanced Topics

### Cutting-Edge Research

#### 1. Adaptive Backdoors
- Triggers that evolve
- Context-aware activation
- Defense-resistant designs

#### 2. Federated Learning Security
- Byzantine-robust aggregation
- Secure multi-party computation
- Privacy-preserving poisoning detection

#### 3. Certified Defenses
- Provable robustness
- Randomized smoothing
- Certified removal

#### 4. Physical-World Attacks
- 3D-printed triggers
- Environmental conditions
- Real-world deployment

## Tools and Frameworks

### Attack Tools
- BackdoorBox
- TrojanZoo
- Custom implementations

### Defense Tools
- Neural Cleanse
- STRIP
- Activation Clustering
- Spectral Signatures

### Analysis Tools
- Model inspection
- Trigger visualization
- Attack success metrics

## Common Pitfalls

1. **Obvious Triggers**
   - Solution: Use subtle, natural triggers

2. **High Poison Rate**
   - Solution: Optimize for minimal poisoning

3. **Accuracy Drop**
   - Solution: Balance attack and utility

4. **Easy Detection**
   - Solution: Test against multiple defenses

5. **Unreliable Activation**
   - Solution: Optimize trigger robustness

## Ethical Considerations

### Research Ethics
- Only test on authorized systems
- Use synthetic/controlled datasets
- Consider dual-use implications
- Responsible disclosure

### Real-World Impact
- Patient safety in healthcare
- Autonomous vehicle security
- Financial system integrity
- National security implications

## Additional Resources

### Research Papers
- "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" (Gu et al., 2017)
- "Trojaning Attack on Neural Networks" (Liu et al., 2018)
- "Neural Cleanse: Identifying and Mitigating Backdoor Attacks" (Wang et al., 2019)
- "Certified Defenses for Data Poisoning Attacks" (Steinhardt et al., 2017)

### Tools
- [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox)
- [TrojanZoo](https://github.com/ain-soph/trojanzoo)
- [ART Poisoning Attacks](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

### Standards
- NIST AI Risk Management Framework
- ISO/IEC 24029 (AI Robustness)
- MITRE ATT&CK for ML

## Next Steps

After completing this module:
1. Review all poisoning techniques
2. Complete detection exercises
3. Design comprehensive defenses
4. Proceed to [Module 6: Advanced LLM Attacks](../06_advanced_attacks/README.md)

---

**Warning**: Poisoning attacks can cause serious harm. Only use these techniques for authorized security testing and research.

# Module 4: Data Extraction & Privacy Attacks

## Overview

This module focuses on privacy attacks against AI/ML systems, including extracting training data, reconstructing sensitive information, and determining dataset membership. These attacks pose serious privacy risks in production systems.

## Learning Objectives

By the end of this module, you will be able to:
- Extract memorized training data from LLMs
- Perform model inversion attacks to reconstruct inputs
- Execute membership inference attacks
- Understand privacy metrics and differential privacy
- Assess privacy risks in deployed models
- Implement privacy-preserving defenses

## Module Contents

### Theory
Comprehensive theory is integrated throughout this README, covering:
- Training data extraction techniques and real-world cases
- Model inversion attack methods and privacy implications
- Membership inference strategies and defense mechanisms
- Privacy metrics, regulatory compliance, and protection strategies

### Labs
1. [Lab 1: Training Data Extraction from LLMs](labs/lab1_training_data_extraction.ipynb)
2. [Lab 2: Membership Inference Attacks](labs/lab2_membership_inference.ipynb)
3. [Lab 3: Model Inversion Techniques](labs/lab3_model_inversion.ipynb)

### Practical Exercises
- Extracting PII from language models
- Determining dataset membership
- Reconstructing training images
- Privacy risk assessment

## Key Concepts

### Training Data Extraction

**What it is**: Recovering verbatim training data from model outputs

**Attack Vectors**:
- Prompt-based extraction
- Completion-based extraction
- Systematic querying
- Temperature manipulation

**Real-World Impact**:
- PII leakage (emails, phone numbers, addresses)
- Proprietary information disclosure
- Copyright violations
- Compliance issues (GDPR, CCPA)

### Model Inversion

**What it is**: Reconstructing input data from model outputs or parameters

**Types**:
- **White-box**: Full model access
- **Black-box**: Query-based reconstruction
- **Gradient-based**: Using gradient information
- **Optimization-based**: Iterative refinement

**Applications**:
- Face reconstruction from embeddings
- Text reconstruction from representations
- Sensitive attribute inference

### Membership Inference

**What it is**: Determining if a specific data point was in the training set

**Attack Methods**:
- **Confidence-based**: Analyzing prediction confidence
- **Loss-based**: Comparing loss values
- **Metric-based**: Using distance metrics
- **Shadow model**: Training auxiliary models

**Privacy Implications**:
- Medical record exposure
- Financial data leakage
- Personal information disclosure
- Dataset composition revelation

## Attack Taxonomy

```
Privacy Attacks
├── Training Data Extraction
│   ├── Prompt-based
│   ├── Completion-based
│   ├── Systematic Querying
│   └── Temperature Manipulation
├── Model Inversion
│   ├── White-box Inversion
│   ├── Black-box Inversion
│   ├── Gradient-based
│   └── Optimization-based
└── Membership Inference
    ├── Confidence-based
    ├── Loss-based
    ├── Metric-based
    └── Shadow Model
```

## Real-World Case Studies

### Case 1: GPT-2 Training Data Extraction (2020)
**Attack**: Researchers extracted memorized training data
**Method**: Systematic prompting with temperature sampling
**Result**: Recovered emails, phone numbers, URLs
**Impact**: Demonstrated privacy risks in large language models
**Source**: Carlini et al. "Extracting Training Data from Large Language Models"

### Case 2: Face Recognition Model Inversion (2015)
**Attack**: Reconstructed faces from model outputs
**Method**: Gradient-based optimization
**Result**: Recognizable face images
**Impact**: Privacy concerns for biometric systems
**Source**: Fredrikson et al. "Model Inversion Attacks that Exploit Confidence Information"

### Case 3: Medical ML Membership Inference (2017)
**Attack**: Determined if patients were in training data
**Method**: Shadow model attack
**Result**: High accuracy membership detection
**Impact**: HIPAA compliance concerns
**Source**: Shokri et al. "Membership Inference Attacks Against Machine Learning Models"

### Case 4: ChatGPT Training Data Leakage (2023)
**Attack**: Extracting training data through crafted prompts
**Method**: Divergence-based extraction
**Result**: Verbatim text from training corpus
**Impact**: Ongoing privacy and copyright concerns
**Source**: Nasr et al. "Scalable Extraction of Training Data from (Production) Language Models"

## Privacy Metrics

### Memorization Metrics
- **Extraction Rate**: Percentage of training data recoverable
- **Exposure**: Amount of sensitive information leaked
- **Verbatim Copying**: Exact matches to training data

### Membership Inference Metrics
- **Attack Accuracy**: Correct membership predictions
- **True Positive Rate**: Correctly identified members
- **False Positive Rate**: Incorrectly identified non-members
- **AUC-ROC**: Overall attack performance

### Privacy Loss Metrics
- **Epsilon (ε)**: Differential privacy parameter
- **Delta (δ)**: Failure probability
- **Privacy Budget**: Cumulative privacy loss

## Defense Mechanisms

### Training-Time Defenses
1. **Differential Privacy**
   - DP-SGD (Differentially Private Stochastic Gradient Descent)
   - Noise injection during training
   - Privacy budget management

2. **Data Sanitization**
   - PII removal from training data
   - Anonymization techniques
   - Data filtering

3. **Regularization**
   - Preventing memorization
   - Early stopping
   - Dropout and weight decay

### Inference-Time Defenses
1. **Output Perturbation**
   - Adding noise to predictions
   - Rounding probabilities
   - Temperature scaling

2. **Query Limiting**
   - Rate limiting
   - Query budgets
   - Anomaly detection

3. **Output Filtering**
   - PII detection and redaction
   - Similarity checks
   - Confidence thresholding

### Architectural Defenses
1. **Federated Learning**
   - Decentralized training
   - Local data privacy
   - Secure aggregation

2. **Secure Enclaves**
   - Trusted execution environments
   - Hardware-based protection
   - Encrypted computation

3. **Knowledge Distillation**
   - Training smaller models
   - Reducing memorization
   - Privacy-preserving transfer

## Privacy-Utility Trade-offs

### Key Considerations
- **Accuracy vs Privacy**: Stronger privacy often reduces accuracy
- **Utility vs Protection**: More protection may limit functionality
- **Cost vs Benefit**: Privacy measures have computational costs

### Balancing Strategies
1. Assess privacy requirements
2. Determine acceptable accuracy loss
3. Implement layered defenses
4. Monitor privacy metrics
5. Adjust based on risk assessment

## Regulatory Compliance

### GDPR (General Data Protection Regulation)
- Right to be forgotten
- Data minimization
- Purpose limitation
- Privacy by design

### CCPA (California Consumer Privacy Act)
- Consumer data rights
- Opt-out mechanisms
- Data disclosure requirements

### HIPAA (Health Insurance Portability and Accountability Act)
- Protected health information
- De-identification requirements
- Security safeguards

### Industry Standards
- NIST Privacy Framework
- ISO/IEC 27701
- IEEE P7002 (Data Privacy Process)

## Prerequisites

- Completion of Modules 1-3
- Understanding of probability and statistics
- Familiarity with privacy concepts
- Python programming proficiency

## Estimated Time

- Theory: 4-5 hours
- Labs: 6-7 hours
- Total: 10-12 hours

## Assessment

### Knowledge Check
- Privacy attack types quiz
- Defense mechanism identification
- Metric calculation exercises
- Compliance requirement matching

### Practical Assessment
1. Extract training data from a provided LLM
2. Perform membership inference on a dataset
3. Reconstruct training samples using model inversion
4. Conduct privacy risk assessment
5. Write a privacy impact report

## Success Criteria

- Successfully extract memorized training data
- Achieve >70% accuracy in membership inference
- Reconstruct recognizable training samples
- Demonstrate understanding of privacy metrics
- Complete privacy risk assessment

## Tools and Frameworks

### Attack Tools
- Custom extraction scripts
- Membership inference frameworks
- Model inversion implementations

### Defense Tools
- Opacus (PyTorch differential privacy)
- TensorFlow Privacy
- PySyft (privacy-preserving ML)

### Analysis Tools
- Privacy metric calculators
- PII detection libraries
- Compliance checking tools

## Common Pitfalls

1. **Underestimating Memorization**
   - Solution: Test thoroughly with diverse prompts

2. **Ignoring Indirect Leakage**
   - Solution: Consider all output channels

3. **Insufficient Privacy Budget**
   - Solution: Carefully manage epsilon values

4. **Over-relying on Single Defense**
   - Solution: Implement layered protection

5. **Not Monitoring Privacy Loss**
   - Solution: Continuous privacy auditing

## Ethical Considerations

### Responsible Disclosure
- Report privacy vulnerabilities appropriately
- Allow time for remediation
- Consider user impact

### Testing Guidelines
- Only test authorized systems
- Use synthetic sensitive data when possible
- Minimize actual privacy harm

### Research Ethics
- IRB approval for human subjects
- Informed consent
- Data protection measures

## Advanced Topics

### Emerging Threats
- Federated learning attacks
- Split learning vulnerabilities
- Encrypted model attacks
- Quantum computing implications

### Cutting-Edge Defenses
- Cryptographic privacy guarantees
- Zero-knowledge proofs
- Homomorphic encryption
- Secure multi-party computation

## Additional Resources

### Research Papers
- "Extracting Training Data from Large Language Models" (Carlini et al., 2021)
- "Membership Inference Attacks Against Machine Learning Models" (Shokri et al., 2017)
- "Model Inversion Attacks that Exploit Confidence Information" (Fredrikson et al., 2015)
- "Deep Learning with Differential Privacy" (Abadi et al., 2016)

### Tools & Libraries
- [Opacus](https://opacus.ai/) - PyTorch differential privacy
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [PySyft](https://github.com/OpenMined/PySyft)
- [Presidio](https://github.com/microsoft/presidio) - PII detection

### Standards & Guidelines
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)
- [OWASP Privacy Risks](https://owasp.org/www-project-top-10-privacy-risks/)
- [ISO/IEC 27701](https://www.iso.org/standard/71670.html)

## Next Steps

After completing this module:
1. Review all lab exercises
2. Complete the privacy assessment
3. Document your findings
4. Proceed to [Module 5: Model Poisoning](../05_poisoning/README.md)

## Lab Overview

### Lab 1: Training Data Extraction
- Extract memorized data from GPT-2
- Test different prompting strategies
- Analyze extraction success rates
- Implement detection mechanisms

### Lab 2: Membership Inference
- Build shadow models
- Execute confidence-based attacks
- Measure attack accuracy
- Analyze privacy leakage

### Lab 3: Model Inversion
- Reconstruct training samples
- Compare white-box vs black-box methods
- Evaluate reconstruction quality
- Test defense effectiveness

## Troubleshooting

### Common Issues

**Issue**: Low extraction success rate
- Try different temperature values
- Use more diverse prompts
- Increase number of attempts

**Issue**: Membership inference fails
- Verify shadow model training
- Check threshold calibration
- Ensure sufficient queries

**Issue**: DP implementation errors
- Verify noise scale calculation
- Check gradient clipping
- Monitor privacy budget

**Issue**: High utility loss with privacy
- Adjust epsilon parameter
- Optimize noise injection
- Consider alternative defenses

---

**Ready to begin?** Start with [Lab 1: Training Data Extraction](labs/lab1_training_data_extraction.ipynb)

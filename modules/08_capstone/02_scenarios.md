# Capstone Project Scenarios

## Overview

This document provides three comprehensive scenarios for your capstone project. Each scenario represents a real-world AI system with unique security challenges. Select the scenario that best aligns with your interests and career goals.

## Scenario Selection Criteria

Consider these factors when selecting your scenario:

**Interest**: Choose a domain that excites you
**Relevance**: Select based on career goals
**Complexity**: Match your skill level
**Resources**: Ensure you have necessary tools
**Time**: Consider available time commitment

## Scenario 1: Enterprise Chatbot Security Assessment

### System Description

**Name**: CustomerAssist AI
**Type**: Enterprise customer service chatbot with RAG
**Technology Stack**:
- LLM: GPT-3.5-turbo or Llama-2-7b
- Vector DB: Pinecone or Chroma
- Framework: LangChain
- Deployment: Cloud-based API

**Functionality**:
- Answer customer questions
- Process returns and refunds
- Provide product recommendations
- Access customer account information
- Escalate to human agents

**Security Controls**:
- Input length limits
- Content filtering
- Rate limiting
- Basic prompt templates
- Logging and monitoring

### Threat Model

**Threat Actors**:
1. **Malicious Customers**: Seeking unauthorized discounts or information
2. **Competitors**: Attempting to extract business intelligence
3. **Attackers**: Testing for data exfiltration
4. **Researchers**: Probing for vulnerabilities

**Attack Scenarios**:
- Prompt injection to bypass authorization
- Jailbreaking to access restricted functions
- Data extraction from vector database
- Social engineering through multi-turn conversations
- Guardrail bypass for inappropriate content

### Testing Objectives

**Primary Objectives**:
1. Test prompt injection resistance
2. Evaluate jailbreaking vulnerabilities
3. Assess data leakage risks
4. Test guardrail effectiveness
5. Evaluate RAG security

**Secondary Objectives**:
- Test multi-turn attack sequences
- Evaluate context window exploitation
- Assess system prompt extraction
- Test function calling security

### Expected Findings

**High-Severity**:
- Unauthorized access to customer data
- Bypass of discount authorization
- Extraction of system prompts
- RAG poisoning vulnerabilities

**Medium-Severity**:
- Guardrail bypass techniques
- Information disclosure
- Context manipulation
- Rate limit bypass

**Low-Severity**:
- Verbose error messages
- Inconsistent responses
- Minor information leakage

### Deliverables

1. **Technical Report** (20 pages)
   - Prompt injection catalog
   - Jailbreaking techniques
   - RAG security analysis
   - Defense recommendations

2. **Attack Demonstrations**
   - Video walkthroughs
   - Proof-of-concept code
   - Test case library

3. **Security Playbook**
   - Input validation rules
   - Prompt engineering guidelines
   - Monitoring strategies
   - Incident response procedures

### Resources

**Tools**:
- LangChain for RAG implementation
- OpenAI or Hugging Face APIs
- Vector database (Chroma/Pinecone)
- Custom attack scripts

**References**:
- OWASP LLM Top 10
- LangChain security documentation
- RAG security research papers

## Scenario 2: Content Moderation System Audit

### System Description

**Name**: SafeContent AI
**Type**: AI-powered content moderation system
**Technology Stack**:
- Model: Fine-tuned BERT or RoBERTa
- Framework: PyTorch
- Deployment: Real-time API
- Scale: 1M+ requests/day

**Functionality**:
- Detect toxic content
- Identify hate speech
- Flag misinformation
- Classify content categories
- Provide confidence scores

**Security Controls**:
- Adversarial training
- Ensemble models
- Confidence thresholds
- Human review queue
- Regular model updates

### Threat Model

**Threat Actors**:
1. **Bad Actors**: Evading moderation to post harmful content
2. **Trolls**: Testing system limits
3. **Adversaries**: Systematic evasion campaigns
4. **Researchers**: Probing for biases

**Attack Scenarios**:
- Character-level evasion (l33t speak)
- Semantic evasion (paraphrasing)
- Adversarial perturbations
- Context manipulation
- Fairness attacks (bias exploitation)

### Testing Objectives

**Primary Objectives**:
1. Test evasion techniques
2. Evaluate adversarial robustness
3. Assess fairness and bias
4. Test detection bypass methods
5. Evaluate false positive rates

**Secondary Objectives**:
- Test transfer attacks
- Evaluate ensemble robustness
- Assess confidence calibration
- Test edge cases

### Expected Findings

**High-Severity**:
- Systematic evasion techniques
- Bias in moderation decisions
- Adversarial example vulnerabilities
- False negative patterns

**Medium-Severity**:
- Character substitution bypass
- Context-dependent failures
- Confidence miscalibration
- Edge case misclassification

**Low-Severity**:
- Minor language variations
- Slang handling issues
- Performance degradation

### Deliverables

1. **Robustness Report** (20 pages)
   - Evasion technique catalog
   - Adversarial robustness analysis
   - Fairness evaluation
   - Defense recommendations

2. **Attack Library**
   - Evasion templates
   - Adversarial examples
   - Test case suite
   - Automated testing scripts

3. **Monitoring Strategy**
   - Detection metrics
   - Drift monitoring
   - Anomaly detection
   - Alert thresholds

### Resources

**Tools**:
- TextAttack for NLP attacks
- ART for adversarial testing
- Fairlearn for bias evaluation
- Custom evasion scripts

**References**:
- Content moderation research
- Adversarial NLP papers
- Fairness in ML literature

## Scenario 3: Medical AI Assistant Privacy Assessment

### System Description

**Name**: MedAssist Pro
**Type**: Healthcare diagnosis support system
**Technology Stack**:
- Model: Fine-tuned medical LLM
- Data: Electronic health records
- Framework: TensorFlow/PyTorch
- Deployment: HIPAA-compliant cloud

**Functionality**:
- Symptom analysis
- Diagnosis suggestions
- Treatment recommendations
- Drug interaction checking
- Medical literature search

**Security Controls**:
- Differential privacy
- Data anonymization
- Access controls
- Audit logging
- Encryption at rest/transit

### Threat Model

**Threat Actors**:
1. **Malicious Insiders**: Attempting data extraction
2. **External Attackers**: Seeking patient information
3. **Competitors**: Model extraction attempts
4. **Researchers**: Privacy testing

**Attack Scenarios**:
- Membership inference attacks
- Model inversion for patient data
- Training data extraction
- Model extraction/stealing
- Privacy leakage through queries

### Testing Objectives

**Primary Objectives**:
1. Test privacy preservation
2. Evaluate membership inference risks
3. Assess model inversion vulnerabilities
4. Test data extraction attacks
5. Evaluate differential privacy effectiveness

**Secondary Objectives**:
- Test model extraction
- Evaluate query-based attacks
- Assess anonymization effectiveness
- Test compliance with HIPAA

### Expected Findings

**Critical-Severity**:
- Patient data leakage
- Membership inference success
- Model inversion revealing PII
- Training data extraction

**High-Severity**:
- Model extraction vulnerabilities
- Anonymization bypass
- Privacy budget exhaustion
- Audit log gaps

**Medium-Severity**:
- Information disclosure
- Confidence leakage
- Metadata exposure

### Deliverables

1. **Privacy Assessment Report** (25 pages)
   - Privacy attack analysis
   - Membership inference results
   - Model inversion findings
   - Compliance evaluation
   - Privacy-preserving recommendations

2. **Risk Analysis**
   - HIPAA compliance gaps
   - Privacy risk matrix
   - Data flow analysis
   - Threat scenarios

3. **Security Architecture Review**
   - Privacy controls evaluation
   - Differential privacy assessment
   - Encryption review
   - Access control analysis

### Resources

**Tools**:
- Privacy Meter for membership inference
- ART for model inversion
- TensorFlow Privacy
- Custom privacy attack scripts

**References**:
- HIPAA security guidelines
- Medical AI privacy research
- Differential privacy papers
- Healthcare security standards

## Scenario Comparison Matrix

| Aspect | Chatbot | Moderation | Medical AI |
|--------|---------|------------|------------|
| **Complexity** | Medium | Medium-High | High |
| **Time Required** | 4-5 weeks | 5-6 weeks | 6+ weeks |
| **Technical Depth** | Moderate | High | Very High |
| **Domain Knowledge** | Low | Medium | High |
| **Privacy Focus** | Medium | Low | Critical |
| **Attack Variety** | High | Medium | Medium |
| **Real-World Impact** | Medium | High | Critical |

## Selection Guidance

### Choose Scenario 1 (Chatbot) if:
- You're interested in prompt engineering
- You want to focus on LLM-specific attacks
- You have limited time (4-5 weeks)
- You're new to security assessments
- You want broad attack coverage

### Choose Scenario 2 (Moderation) if:
- You're interested in adversarial ML
- You want to focus on evasion techniques
- You care about fairness and bias
- You have NLP experience
- You want to build attack libraries

### Choose Scenario 3 (Medical AI) if:
- You're interested in privacy
- You have healthcare domain knowledge
- You want maximum challenge
- You have 6+ weeks available
- You care about compliance

## Custom Scenario Option

If none of these scenarios fit your needs, you can propose a custom scenario with instructor approval.

**Requirements**:
- Clear system description
- Defined threat model
- Specific testing objectives
- Realistic scope
- Available resources

**Proposal Template**:
1. System overview
2. Security concerns
3. Testing approach
4. Expected outcomes
5. Resource requirements

## Next Steps

1. Review all three scenarios carefully
2. Assess your interests and constraints
3. Select your scenario
4. Complete the project charter
5. Begin reconnaissance phase

---

**Ready to select your scenario? Proceed to the capstone notebook to begin!**

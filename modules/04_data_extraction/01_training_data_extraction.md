# Training Data Extraction

## Introduction

Training data extraction is the process of recovering verbatim or near-verbatim training data from a trained machine learning model. This is a critical privacy concern, especially for large language models that may have been trained on sensitive or proprietary data.

## Why Models Memorize

### Memorization vs Generalization

**Generalization**: Learning patterns that apply to unseen data
```
Training: "The cat sat on the mat"
Generalization: Understanding subject-verb-object structure
```

**Memorization**: Storing specific training examples
```
Training: "John Smith's email is john.smith@example.com"
Memorization: Exact reproduction of this string
```

### Factors Affecting Memorization

1. **Data Duplication**
   - Repeated examples are more likely to be memorized
   - Common in web-scraped datasets
   - Increases memorization risk

2. **Model Capacity**
   - Larger models can memorize more
   - Overparameterization enables memorization
   - Trade-off with generalization

3. **Training Duration**
   - Longer training increases memorization
   - Overfitting to training data
   - Early stopping can help

4. **Data Rarity**
   - Unique or rare examples more memorable
   - Outliers in the distribution
   - Sensitive data often rare

5. **Sequence Length**
   - Longer sequences more likely memorized
   - Especially for exact matches
   - Context-dependent memorization

## Attack Methodologies

### 1. Prompt-Based Extraction

**Basic Approach**: Use prompts to trigger memorized content

```python
# Example prompts
prompts = [
    "My email address is",
    "You can reach me at",
    "Contact information:",
    "Phone number:",
]
```

**Advanced Techniques**:
- Context priming
- Completion forcing
- Template-based extraction

### 2. Temperature Sampling

**Concept**: Higher temperature increases diversity, may reveal memorization

```python
# Low temperature (focused)
response = model.generate(prompt, temperature=0.1)

# High temperature (diverse)
response = model.generate(prompt, temperature=1.5)
```

**Strategy**:
- Generate multiple samples
- Look for repeated exact matches
- Indicates memorization

### 3. Prefix-Based Extraction

**Method**: Provide known prefixes to extract continuations

```python
# Known prefix from training data
prefix = "The quick brown fox"

# Extract continuation
continuation = model.generate(prefix, max_length=100)
```

**Applications**:
- Extracting copyrighted text
- Recovering code snippets
- Finding PII

### 4. Divergence-Based Extraction

**Technique**: Exploit model's tendency to diverge from typical outputs

```python
# Generate many samples
samples = [model.generate(prompt) for _ in range(1000)]

# Find outliers (potential memorization)
outliers = find_divergent_samples(samples)
```

**Indicators**:
- Unusual perplexity
- High confidence on rare sequences
- Exact repetition across samples

## Extraction Strategies

### Strategy 1: Systematic Prompting

```python
def systematic_extraction(model, prompt_templates, num_samples=100):
    """
    Systematically test prompts for memorization.
    """
    extracted = []
    
    for template in prompt_templates:
        for i in range(num_samples):
            # Generate with high temperature
            output = model.generate(
                template,
                temperature=1.5,
                max_length=100
            )
            
            # Check for PII patterns
            if contains_pii(output):
                extracted.append(output)
    
    return extracted
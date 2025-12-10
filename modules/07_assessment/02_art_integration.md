# Adversarial Robustness Toolbox (ART) Integration

## Introduction

The Adversarial Robustness Toolbox (ART) is a comprehensive Python library for machine learning security. Developed by IBM Research, ART provides tools for adversarial attacks, defenses, and robustness evaluation across multiple ML frameworks.

## ART Architecture

### Core Components

```
ART Architecture
├── Estimators (Model Wrappers)
│   ├── Classifiers
│   ├── Regressors
│   ├── Object Detectors
│   └── Generators
├── Attacks
│   ├── Evasion
│   ├── Poisoning
│   ├── Extraction
│   └── Inference
├── Defenses
│   ├── Preprocessors
│   ├── Postprocessors
│   └── Trainers
└── Metrics
    ├── Robustness
    ├── Privacy
    └── Verification
```

### Estimators

Estimators wrap ML models to provide a unified interface for attacks and defenses.

**Supported Frameworks**:
- PyTorch
- TensorFlow/Keras
- scikit-learn
- XGBoost
- LightGBM
- CatBoost

**Example**:
```python
from art.estimators.classification import PyTorchClassifier

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=10,
    clip_values=(0, 1)
)
```

### Attack Categories

#### 1. Evasion Attacks
**Purpose**: Manipulate inputs to cause misclassification

**White-box Attacks**:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Carlini & Wagner (C&W)
- DeepFool
- Elastic Net Attack

**Black-box Attacks**:
- HopSkipJump
- Boundary Attack
- ZOO (Zeroth Order Optimization)
- Square Attack

**Example**:
```python
from art.attacks.evasion import ProjectedGradientDescent

attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.3,
    eps_step=0.01,
    max_iter=100,
    targeted=False
)

x_adv = attack.generate(x=x_test)
```

#### 2. Poisoning Attacks
**Purpose**: Corrupt training data to compromise model

**Attack Types**:
- Clean-label poisoning
- Backdoor attacks
- Feature collision
- Gradient matching

**Example**:
```python
from art.attacks.poisoning import PoisoningAttackBackdoor

backdoor = PoisoningAttackBackdoor(
    perturbation=backdoor_pattern
)

x_poisoned, y_poisoned = backdoor.poison(x_train, y_train)
```

#### 3. Extraction Attacks
**Purpose**: Steal model functionality or parameters

**Attack Types**:
- Copycat CNN
- Knockoff Nets
- Functionally Equivalent Extraction

**Example**:
```python
from art.attacks.extraction import CopycatCNN

extraction = CopycatCNN(
    classifier=victim_classifier,
    batch_size_fit=128,
    batch_size_query=128,
    nb_epochs=10,
    nb_stolen=10000
)

stolen_classifier = extraction.extract(x=x_query)
```

#### 4. Inference Attacks
**Purpose**: Infer sensitive information about training data

**Attack Types**:
- Membership inference
- Attribute inference
- Model inversion

**Example**:
```python
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

mia = MembershipInferenceBlackBox(
    classifier,
    attack_model_type='nn'
)

inferred_train = mia.infer(x_test, y_test)
```

### Defense Mechanisms

#### 1. Preprocessors
**Purpose**: Transform inputs before model processing

**Defense Types**:
- Gaussian augmentation
- JPEG compression
- Spatial smoothing
- Feature squeezing
- Thermometer encoding

**Example**:
```python
from art.defenses.preprocessor import GaussianAugmentation

defense = GaussianAugmentation(sigma=0.1, augmentation=True)
x_defended = defense(x_test)
```

#### 2. Postprocessors
**Purpose**: Transform model outputs

**Defense Types**:
- High confidence
- Reverse sigmoid
- Gaussian noise

**Example**:
```python
from art.defenses.postprocessor import HighConfidence

postprocessor = HighConfidence(cutoff=0.9)
predictions_defended = postprocessor(predictions)
```

#### 3. Adversarial Training
**Purpose**: Train models on adversarial examples

**Example**:
```python
from art.defenses.trainer import AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescent

attack = ProjectedGradientDescent(classifier, eps=0.3)
trainer = AdversarialTrainer(classifier, attacks=attack)

trainer.fit(x_train, y_train, nb_epochs=10)
```

## Practical Integration

### Setup and Configuration

```python
# Installation
# pip install adversarial-robustness-toolbox

# Basic imports
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defenses.preprocessor import GaussianAugmentation
from art.utils import load_mnist

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
```

### Model Wrapping

#### PyTorch Models
```python
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

# Define model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(5408, 10)
)

# Wrap with ART
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0, 1)
)
```

#### TensorFlow/Keras Models
```python
from tensorflow import keras
from art.estimators.classification import KerasClassifier

# Define model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Wrap with ART
classifier = KerasClassifier(model=model, clip_values=(0, 1))
```

#### Hugging Face Transformers
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn

# Load model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom wrapper for text models
class TextClassifierWrapper(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def forward(self, x):
        # x is already tokenized
        return self.model(x).logits

wrapper = TextClassifierWrapper(model, tokenizer)

classifier = PyTorchClassifier(
    model=wrapper,
    loss=nn.CrossEntropyLoss(),
    input_shape=(512,),  # max sequence length
    nb_classes=2,
    clip_values=(0, 1)
)
```

### Attack Execution

#### Single Attack
```python
from art.attacks.evasion import ProjectedGradientDescent

# Configure attack
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.3,              # Maximum perturbation
    eps_step=0.01,        # Step size
    max_iter=100,         # Maximum iterations
    targeted=False,       # Untargeted attack
    num_random_init=5,    # Random restarts
    batch_size=128        # Batch size
)

# Generate adversarial examples
x_adv = attack.generate(x=x_test)

# Evaluate
predictions_clean = classifier.predict(x_test)
predictions_adv = classifier.predict(x_adv)

accuracy_clean = np.mean(np.argmax(predictions_clean, axis=1) == y_test)
accuracy_adv = np.mean(np.argmax(predictions_adv, axis=1) == y_test)

print(f"Clean accuracy: {accuracy_clean:.2%}")
print(f"Adversarial accuracy: {accuracy_adv:.2%}")
```

#### Multiple Attacks
```python
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    DeepFool
)

attacks = {
    'FGSM': FastGradientMethod(classifier, eps=0.3),
    'PGD': ProjectedGradientDescent(classifier, eps=0.3),
    'C&W': CarliniL2Method(classifier, confidence=0.0),
    'DeepFool': DeepFool(classifier)
}

results = {}
for name, attack in attacks.items():
    x_adv = attack.generate(x=x_test[:100])  # Test on subset
    predictions = classifier.predict(x_adv)
    accuracy = np.mean(np.argmax(predictions, axis=1) == y_test[:100])
    results[name] = accuracy
    print(f"{name} accuracy: {accuracy:.2%}")
```

### Defense Evaluation

```python
from art.defenses.preprocessor import GaussianAugmentation
from art.defenses.trainer import AdversarialTrainer

# Test preprocessor defense
defense = GaussianAugmentation(sigma=0.1)
x_defended = defense(x_test)[0]

predictions_defended = classifier.predict(x_defended)
accuracy_defended = np.mean(
    np.argmax(predictions_defended, axis=1) == y_test
)

print(f"Defended accuracy: {accuracy_defended:.2%}")

# Adversarial training
attack_for_training = ProjectedGradientDescent(
    classifier, eps=0.3, eps_step=0.01, max_iter=40
)

trainer = AdversarialTrainer(classifier, attacks=attack_for_training)
trainer.fit(x_train, y_train, nb_epochs=10, batch_size=128)

# Re-evaluate
predictions_after_training = classifier.predict(x_adv)
accuracy_after_training = np.mean(
    np.argmax(predictions_after_training, axis=1) == y_test
)

print(f"Accuracy after adversarial training: {accuracy_after_training:.2%}")
```

## Advanced Features

### Custom Attacks

```python
from art.attacks.attack import EvasionAttack

class CustomAttack(EvasionAttack):
    attack_params = ['eps', 'max_iter']
    
    def __init__(self, estimator, eps=0.3, max_iter=100):
        super().__init__(estimator=estimator)
        self.eps = eps
        self.max_iter = max_iter
    
    def generate(self, x, y=None, **kwargs):
        # Implement custom attack logic
        x_adv = x.copy()
        
        for i in range(self.max_iter):
            # Custom perturbation logic
            gradients = self.estimator.loss_gradient(x_adv, y)
            x_adv = x_adv + self.eps * np.sign(gradients)
            x_adv = np.clip(x_adv, 0, 1)
        
        return x_adv
```

### Robustness Metrics

```python
from art.metrics import empirical_robustness

# Calculate empirical robustness
robustness = empirical_robustness(
    classifier,
    x_test,
    attack_name='pgd',
    attack_params={'eps': 0.3}
)

print(f"Empirical robustness: {robustness:.2%}")
```

### Certified Defenses

```python
from art.defenses.preprocessor import PixelDefend
from art.estimators.certification import RandomizedSmoothing

# Randomized smoothing for certified robustness
smoothed_classifier = RandomizedSmoothing(
    classifier=classifier,
    sample_size=100,
    scale=0.1,
    alpha=0.001
)

# Get certified predictions
predictions, radius = smoothed_classifier.certify(x_test, n=1000)
```

## Performance Optimization

### Batch Processing
```python
# Process in batches for memory efficiency
batch_size = 128
x_adv_batches = []

for i in range(0, len(x_test), batch_size):
    batch = x_test[i:i+batch_size]
    x_adv_batch = attack.generate(x=batch)
    x_adv_batches.append(x_adv_batch)

x_adv = np.concatenate(x_adv_batches)
```

### GPU Acceleration
```python
import torch

# Ensure model is on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ART automatically uses GPU if model is on GPU
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=10,
    device_type='gpu'  # Explicitly specify GPU
)
```

### Parallel Execution
```python
from joblib import Parallel, delayed

def attack_sample(x_sample, attack):
    return attack.generate(x=x_sample.reshape(1, *x_sample.shape))

# Parallel attack generation (for black-box attacks)
x_adv_list = Parallel(n_jobs=-1)(
    delayed(attack_sample)(x, attack) for x in x_test
)
x_adv = np.concatenate(x_adv_list)
```

## Best Practices

### 1. Model Wrapping
- Always specify `clip_values` to constrain inputs
- Set correct `input_shape` and `nb_classes`
- Provide loss function and optimizer for gradient-based attacks

### 2. Attack Configuration
- Start with small `eps` values and increase gradually
- Use `batch_size` to manage memory
- Set `max_iter` based on available compute
- Use `num_random_init` for stronger attacks

### 3. Defense Evaluation
- Test multiple attack types
- Vary attack parameters
- Measure both clean and adversarial accuracy
- Consider computational overhead

### 4. Production Deployment
- Cache model predictions when possible
- Use batch processing for efficiency
- Monitor attack detection metrics
- Implement fallback mechanisms

## Common Issues and Solutions

### Issue: Out of Memory
**Solution**: Reduce batch size, process in smaller chunks, use gradient checkpointing

### Issue: Slow Attack Generation
**Solution**: Use GPU, reduce max_iter, try faster attacks (FGSM instead of PGD)

### Issue: Low Attack Success Rate
**Solution**: Increase eps, increase max_iter, try different attacks, check model robustness

### Issue: Framework Compatibility
**Solution**: Check ART version compatibility, update frameworks, use appropriate estimator

## Integration with CI/CD

```python
# Example: Automated robustness testing
def test_model_robustness(model, x_test, y_test, threshold=0.5):
    """
    Test if model meets robustness threshold.
    """
    classifier = PyTorchClassifier(model=model, ...)
    
    attack = ProjectedGradientDescent(classifier, eps=0.3)
    x_adv = attack.generate(x=x_test)
    
    predictions = classifier.predict(x_adv)
    accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
    
    assert accuracy >= threshold, f"Robustness test failed: {accuracy:.2%} < {threshold:.2%}"
    
    return accuracy

# Use in pytest
def test_production_model():
    model = load_production_model()
    x_test, y_test = load_test_data()
    
    robustness = test_model_robustness(model, x_test, y_test, threshold=0.6)
    print(f"Model robustness: {robustness:.2%}")
```

## Conclusion

ART provides a comprehensive toolkit for AI security assessment. Its framework-agnostic design, extensive attack library, and defense mechanisms make it ideal for enterprise security testing. Master ART integration to build robust, automated security pipelines.

## Key Takeaways

1. **Unified Interface**: ART provides consistent API across frameworks
2. **Comprehensive Coverage**: 50+ attacks, 20+ defenses
3. **Production-Ready**: Suitable for enterprise deployment
4. **Active Development**: Regular updates and new features
5. **Well-Documented**: Extensive documentation and examples

## Next Steps

1. Complete Lab 1 for hands-on ART experience
2. Experiment with different attack types
3. Implement custom attacks for specific use cases
4. Integrate ART into your security pipeline

---

**Ready to master ART? Proceed to Lab 1!**

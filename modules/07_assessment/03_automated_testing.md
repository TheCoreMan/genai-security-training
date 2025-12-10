# Automated Testing and Continuous Security

## Introduction

Automated security testing is essential for maintaining AI system security at scale. This document covers strategies for integrating security testing into CI/CD pipelines, implementing continuous monitoring, and building automated assessment workflows.

## Why Automate Security Testing?

### Benefits

**Consistency**: Automated tests run the same way every time
**Speed**: Rapid feedback on security posture
**Coverage**: Test more scenarios than manual testing
**Regression Prevention**: Catch security regressions early
**Scalability**: Test multiple models simultaneously
**Cost Efficiency**: Reduce manual testing overhead

### Challenges

**Initial Setup**: Requires upfront investment
**Maintenance**: Tests need updates as attacks evolve
**False Positives**: May flag benign changes
**Compute Resources**: Security tests can be resource-intensive
**Complexity**: Integration with existing pipelines

## Automated Testing Architecture

```
Automated Security Testing Pipeline
├── Trigger Events
│   ├── Code commits
│   ├── Model updates
│   ├── Scheduled runs
│   └── Manual triggers
├── Test Execution
│   ├── Unit tests (individual attacks)
│   ├── Integration tests (full pipeline)
│   ├── Regression tests (known vulnerabilities)
│   └── Exploratory tests (new attacks)
├── Result Analysis
│   ├── Metric computation
│   ├── Threshold checking
│   ├── Trend analysis
│   └── Anomaly detection
└── Reporting & Actions
    ├── Test reports
    ├── Alerts & notifications
    ├── Automated remediation
    └── Documentation updates
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/security-testing.yml
name: AI Security Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  security-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install adversarial-robustness-toolbox textattack
    
    - name: Run adversarial robustness tests
      run: |
        python tests/test_robustness.py --model-path models/latest.pt
    
    - name: Run prompt injection tests
      run: |
        python tests/test_prompt_injection.py
    
    - name: Run data extraction tests
      run: |
        python tests/test_data_extraction.py
    
    - name: Generate security report
      run: |
        python scripts/generate_security_report.py
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: security-test-results
        path: reports/
    
    - name: Check security thresholds
      run: |
        python scripts/check_thresholds.py --fail-on-violation
```

### GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - test
  - security
  - report

security_tests:
  stage: security
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python tests/run_security_suite.py
  artifacts:
    reports:
      junit: reports/security-tests.xml
    paths:
      - reports/
  only:
    - main
    - merge_requests

robustness_check:
  stage: security
  script:
    - python tests/test_adversarial_robustness.py
  allow_failure: false
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
```

### Jenkins Pipeline Example

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Security Tests') {
            parallel {
                stage('Adversarial Tests') {
                    steps {
                        sh 'python tests/test_adversarial.py'
                    }
                }
                stage('Prompt Injection Tests') {
                    steps {
                        sh 'python tests/test_prompt_injection.py'
                    }
                }
                stage('Privacy Tests') {
                    steps {
                        sh 'python tests/test_privacy.py'
                    }
                }
            }
        }
        
        stage('Generate Report') {
            steps {
                sh 'python scripts/generate_report.py'
                publishHTML([
                    reportDir: 'reports',
                    reportFiles: 'security_report.html',
                    reportName: 'Security Report'
                ])
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    def threshold = 0.7
                    def robustness = readFile('reports/robustness_score.txt').trim().toFloat()
                    if (robustness < threshold) {
                        error("Robustness score ${robustness} below threshold ${threshold}")
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'reports/**/*', fingerprint: true
        }
        failure {
            mail to: 'security-team@example.com',
                 subject: "Security Test Failed: ${env.JOB_NAME}",
                 body: "Security tests failed. Check ${env.BUILD_URL}"
        }
    }
}
```

## Test Implementation

### Unit Test Example

```python
# tests/test_robustness.py
import pytest
import torch
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

class TestAdversarialRobustness:
    @pytest.fixture
    def classifier(self):
        """Load model and wrap with ART."""
        model = torch.load('models/latest.pt')
        return PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=10
        )
    
    @pytest.fixture
    def test_data(self):
        """Load test dataset."""
        return torch.load('data/test_set.pt')
    
    def test_fgsm_robustness(self, classifier, test_data):
        """Test robustness against FGSM attack."""
        from art.attacks.evasion import FastGradientMethod
        
        x_test, y_test = test_data
        attack = FastGradientMethod(classifier, eps=0.3)
        x_adv = attack.generate(x=x_test)
        
        predictions = classifier.predict(x_adv)
        accuracy = (predictions.argmax(axis=1) == y_test).mean()
        
        assert accuracy >= 0.6, f"FGSM robustness too low: {accuracy:.2%}"
    
    def test_pgd_robustness(self, classifier, test_data):
        """Test robustness against PGD attack."""
        x_test, y_test = test_data
        attack = ProjectedGradientDescent(
            classifier, eps=0.3, eps_step=0.01, max_iter=40
        )
        x_adv = attack.generate(x=x_test[:100])  # Subset for speed
        
        predictions = classifier.predict(x_adv)
        accuracy = (predictions.argmax(axis=1) == y_test[:100]).mean()
        
        assert accuracy >= 0.5, f"PGD robustness too low: {accuracy:.2%}"
    
    @pytest.mark.parametrize("eps", [0.1, 0.2, 0.3])
    def test_robustness_at_epsilon(self, classifier, test_data, eps):
        """Test robustness at different epsilon values."""
        x_test, y_test = test_data
        attack = ProjectedGradientDescent(classifier, eps=eps)
        x_adv = attack.generate(x=x_test[:50])
        
        predictions = classifier.predict(x_adv)
        accuracy = (predictions.argmax(axis=1) == y_test[:50]).mean()
        
        # Threshold decreases with epsilon
        threshold = max(0.3, 0.8 - eps)
        assert accuracy >= threshold, \
            f"Robustness at ε={eps} too low: {accuracy:.2%}"
```

### Integration Test Example

```python
# tests/test_security_pipeline.py
import pytest
from security_pipeline import SecurityAssessment

class TestSecurityPipeline:
    def test_full_assessment(self):
        """Test complete security assessment pipeline."""
        assessment = SecurityAssessment(
            model_path='models/latest.pt',
            test_data_path='data/test_set.pt'
        )
        
        # Run all tests
        results = assessment.run_all_tests()
        
        # Check all categories
        assert results['adversarial_robustness'] >= 0.6
        assert results['prompt_injection_resistance'] >= 0.7
        assert results['privacy_score'] >= 0.8
        assert results['fairness_score'] >= 0.75
        
        # Generate report
        report = assessment.generate_report()
        assert report is not None
        assert 'vulnerabilities' in report
        assert 'recommendations' in report
```

### Regression Test Example

```python
# tests/test_regression.py
import pytest
import json

class TestSecurityRegression:
    @pytest.fixture
    def baseline_results(self):
        """Load baseline security metrics."""
        with open('baselines/security_metrics.json') as f:
            return json.load(f)
    
    def test_no_regression(self, baseline_results):
        """Ensure security hasn't regressed."""
        from security_pipeline import SecurityAssessment
        
        assessment = SecurityAssessment(model_path='models/latest.pt')
        current_results = assessment.run_all_tests()
        
        for metric, baseline_value in baseline_results.items():
            current_value = current_results[metric]
            
            # Allow 5% degradation
            threshold = baseline_value * 0.95
            
            assert current_value >= threshold, \
                f"{metric} regressed: {current_value:.3f} < {threshold:.3f}"
```

## Continuous Monitoring

### Production Monitoring

```python
# monitoring/security_monitor.py
import time
from prometheus_client import Counter, Histogram, Gauge
from art.attacks.evasion import FastGradientMethod

# Metrics
attack_attempts = Counter('attack_attempts_total', 'Total attack attempts')
attack_success = Counter('attack_success_total', 'Successful attacks')
robustness_score = Gauge('model_robustness_score', 'Current robustness score')
inference_time = Histogram('inference_time_seconds', 'Inference time')

class ProductionSecurityMonitor:
    def __init__(self, model, attack_detector):
        self.model = model
        self.attack_detector = attack_detector
        self.baseline_performance = self.measure_baseline()
    
    def measure_baseline(self):
        """Establish baseline performance metrics."""
        # Run on clean test set
        pass
    
    def monitor_request(self, input_data):
        """Monitor individual prediction request."""
        start_time = time.time()
        
        # Check for attack patterns
        is_attack = self.attack_detector.detect(input_data)
        if is_attack:
            attack_attempts.inc()
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Record metrics
        inference_time.observe(time.time() - start_time)
        
        # Check prediction confidence
        confidence = prediction.max()
        if confidence < 0.5:
            # Potential adversarial example
            attack_success.inc()
        
        return prediction
    
    def periodic_assessment(self):
        """Run periodic robustness assessment."""
        from art.estimators.classification import PyTorchClassifier
        
        classifier = PyTorchClassifier(model=self.model, ...)
        attack = FastGradientMethod(classifier, eps=0.3)
        
        # Test on recent data
        x_test = self.get_recent_samples()
        x_adv = attack.generate(x=x_test)
        
        predictions = classifier.predict(x_adv)
        accuracy = (predictions.argmax(axis=1) == y_test).mean()
        
        robustness_score.set(accuracy)
        
        if accuracy < 0.6:
            self.alert_security_team(f"Robustness dropped to {accuracy:.2%}")
```

### Alerting System

```python
# monitoring/alerting.py
import smtplib
from email.mime.text import MIMEText
from slack_sdk import WebClient

class SecurityAlerting:
    def __init__(self, config):
        self.config = config
        self.slack_client = WebClient(token=config['slack_token'])
    
    def send_alert(self, severity, message, details=None):
        """Send security alert through multiple channels."""
        if severity == 'critical':
            self.send_email(message, details)
            self.send_slack(message, details)
            self.create_incident(message, details)
        elif severity == 'high':
            self.send_slack(message, details)
        else:
            self.log_alert(message, details)
    
    def send_email(self, message, details):
        """Send email alert."""
        msg = MIMEText(f"{message}\n\nDetails:\n{details}")
        msg['Subject'] = f'[SECURITY ALERT] {message}'
        msg['From'] = self.config['from_email']
        msg['To'] = self.config['security_team_email']
        
        with smtplib.SMTP(self.config['smtp_server']) as server:
            server.send_message(msg)
    
    def send_slack(self, message, details):
        """Send Slack notification."""
        self.slack_client.chat_postMessage(
            channel=self.config['slack_channel'],
            text=f":rotating_light: *Security Alert*\n{message}",
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{message}*"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{details}```"}
                }
            ]
        )
```

## Report Generation

### Automated Report Template

```python
# reporting/security_report.py
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SecurityReportGenerator:
    def __init__(self, results):
        self.results = results
        self.timestamp = datetime.now()
    
    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Security Assessment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; }
                .metric { display: inline-block; margin: 20px; }
                .pass { color: green; }
                .fail { color: red; }
                .warning { color: orange; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Security Assessment Report</h1>
                <p>Generated: {{ timestamp }}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <p>Overall Security Score: <strong>{{ overall_score }}</strong></p>
            
            <h2>Test Results</h2>
            {% for test, result in results.items() %}
            <div class="metric">
                <h3>{{ test }}</h3>
                <p class="{{ result.status }}">
                    Score: {{ result.score }}<br>
                    Status: {{ result.status }}
                </p>
            </div>
            {% endfor %}
            
            <h2>Vulnerabilities</h2>
            <ul>
            {% for vuln in vulnerabilities %}
                <li><strong>{{ vuln.severity }}</strong>: {{ vuln.description }}</li>
            {% endfor %}
            </ul>
            
            <h2>Recommendations</h2>
            <ol>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ol>
        </body>
        </html>
        ''')
        
        return template.render(
            timestamp=self.timestamp,
            overall_score=self.calculate_overall_score(),
            results=self.results,
            vulnerabilities=self.identify_vulnerabilities(),
            recommendations=self.generate_recommendations()
        )
    
    def generate_visualizations(self):
        """Generate security metric visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Robustness over time
        axes[0, 0].plot(self.results['robustness_history'])
        axes[0, 0].set_title('Robustness Score Over Time')
        axes[0, 0].set_xlabel('Test Run')
        axes[0, 0].set_ylabel('Score')
        
        # Attack success rates
        attack_types = list(self.results['attack_results'].keys())
        success_rates = [r['success_rate'] for r in self.results['attack_results'].values()]
        axes[0, 1].bar(attack_types, success_rates)
        axes[0, 1].set_title('Attack Success Rates')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Vulnerability distribution
        vuln_counts = self.results['vulnerability_counts']
        axes[1, 0].pie(vuln_counts.values(), labels=vuln_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Vulnerability Distribution')
        
        # Security metrics radar
        categories = list(self.results['security_metrics'].keys())
        values = list(self.results['security_metrics'].values())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        axes[1, 1] = plt.subplot(224, projection='polar')
        axes[1, 1].plot(angles, values)
        axes[1, 1].fill(angles, values, alpha=0.25)
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].set_title('Security Metrics')
        
        plt.tight_layout()
        plt.savefig('reports/security_metrics.png', dpi=300)
```

## Best Practices

### Test Design
1. **Start Simple**: Begin with basic tests, add complexity gradually
2. **Fast Feedback**: Prioritize quick tests for CI, detailed tests for nightly runs
3. **Meaningful Thresholds**: Set realistic thresholds based on business requirements
4. **Comprehensive Coverage**: Test all attack categories and scenarios

### Pipeline Integration
1. **Fail Fast**: Run quick tests first, detailed tests later
2. **Parallel Execution**: Run independent tests in parallel
3. **Resource Management**: Use appropriate compute resources for each test
4. **Artifact Storage**: Save test results and reports for analysis

### Monitoring
1. **Real-time Alerts**: Immediate notification of critical issues
2. **Trend Analysis**: Track metrics over time to identify degradation
3. **Anomaly Detection**: Automatically detect unusual patterns
4. **Regular Audits**: Periodic comprehensive assessments

### Reporting
1. **Actionable Insights**: Provide clear recommendations
2. **Multiple Audiences**: Technical details for engineers, summaries for executives
3. **Visualization**: Use charts and graphs for clarity
4. **Historical Tracking**: Show trends and improvements over time

## Conclusion

Automated security testing is essential for maintaining AI system security at scale. By integrating security tests into CI/CD pipelines, implementing continuous monitoring, and generating automated reports, organizations can proactively identify and address vulnerabilities.

## Key Takeaways

1. **Automation is Essential**: Manual testing doesn't scale
2. **Continuous Testing**: Security is an ongoing process
3. **Multiple Layers**: Combine unit, integration, and regression tests
4. **Actionable Reports**: Focus on insights and recommendations
5. **Iterative Improvement**: Continuously refine tests and thresholds

## Next Steps

1. Complete Lab 4 for hands-on automated testing experience
2. Integrate security tests into your CI/CD pipeline
3. Set up continuous monitoring for production models
4. Build custom reporting dashboards

---

**Ready to automate your security testing? Let's build it in Lab 4!**

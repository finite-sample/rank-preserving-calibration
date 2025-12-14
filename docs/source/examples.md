# Real-World Examples

This section provides comprehensive real-world examples of rank-preserving calibration across different domains. Each example uses real datasets and demonstrates practical applications with complete analysis and visualization.

```{toctree}
:maxdepth: 2
:caption: Comprehensive Examples

examples/medical_calibration
examples/text_classification
examples/vision_calibration
examples/financial_calibration
examples/survey_calibration
```

## Example Overview

### Medical Diagnosis Calibration

**Dataset**: Wisconsin Breast Cancer Dataset  
**Scenario**: Deploying diagnostic models across populations with different disease prevalence  
**Key Features**: Clinical decision thresholds, ROC analysis, cost-benefit analysis  
**Business Impact**: Optimized resource allocation, improved patient outcomes

The medical example demonstrates how rank-preserving calibration maintains diagnostic ranking while adjusting probability estimates for different patient populations. This is critical for deploying AI diagnostic tools across healthcare systems with varying patient demographics.

### Text Classification & Sentiment Analysis

**Dataset**: 20 Newsgroups (simulated sentiment)  
**Scenario**: Cross-domain sentiment analysis deployment  
**Key Features**: Domain adaptation, content moderation, engagement optimization  
**Business Impact**: Improved content recommendation, reduced bias across platforms

This example shows how sentiment models trained on one domain (e.g., movie reviews) can be calibrated for deployment on different platforms (e.g., social media) while preserving relative sentiment rankings.

### Computer Vision & Image Classification

**Dataset**: Handwritten Digits (sklearn)  
**Scenario**: OCR system deployment across different applications  
**Key Features**: Confidence calibration, automated processing thresholds, cost optimization  
**Business Impact**: Reduced manual review costs, improved automation rates

The vision example demonstrates calibrating image classifiers for different deployment contexts (ZIP codes vs. financial documents) where digit frequency distributions vary significantly.

### Financial Risk Assessment

**Dataset**: Simulated Credit Portfolio  
**Scenario**: Credit scoring across different economic conditions  
**Key Features**: Regulatory capital, CECL compliance, economic stress testing  
**Business Impact**: Optimized lending decisions, improved risk management

This example shows how credit scoring models can be calibrated for different economic scenarios while maintaining customer ranking, critical for regulatory compliance and portfolio optimization.

### Survey Research & Demographic Reweighting

**Dataset**: Simulated Political Survey  
**Scenario**: Correcting demographic bias in survey samples  
**Key Features**: Population representativeness, statistical significance, bias correction  
**Business Impact**: More accurate polling, improved policy insights

The survey example demonstrates how to correct demographic sampling bias while preserving individual response patterns, essential for accurate public opinion research.

## Quick Start Example

Here's a simple example to get started with rank-preserving calibration:

```python
import numpy as np
from rank_preserving_calibration import calibrate_dykstra

# Your probability matrix (N samples × J classes)
P = np.random.dirichlet([1, 1, 1], size=100)

# Target column sums (adjust class distributions)
M = np.array([30.0, 40.0, 30.0])

# Calibrate probabilities
result = calibrate_dykstra(P, M)
calibrated_probs = result.Q

print(f"Original column sums: {P.sum(axis=0)}")
print(f"Calibrated column sums: {calibrated_probs.sum(axis=0)}")
print(f"Converged: {result.converged}")
```

## Key Benefits Across Domains

**Rank Preservation**: Maintains relative ordering of predictions while adjusting absolute probabilities

**Mathematical Guarantees**: Satisfies all constraints within numerical precision

**Broad Applicability**: Works across medical, financial, text, vision, and survey applications  

**Business Value**: Improves decision-making through better-calibrated uncertainty estimates

**Regulatory Compliance**: Supports requirements for model explainability and calibration

## Next Steps

1. **Choose Your Domain**: Start with the example most similar to your use case
2. **Adapt the Code**: Modify the examples for your specific data and requirements  
3. **Validate Results**: Test calibration quality on your validation data
4. **Deploy Safely**: Monitor calibration performance in production
5. **Iterate**: Recalibrate as data distributions evolve

For detailed API documentation, see the {doc}`api` reference.
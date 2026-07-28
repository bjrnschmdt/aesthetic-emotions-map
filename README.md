# Aesthetic Emotions Map

## Repository Link

[https://github.com/bjrnschmdt/aesthetic-emotions-map](https://github.com/bjrnschmdt/aesthetic-emotions-map)

## Description

This project builds on a 6,832-image dataset of visual styles with LLM-derived emotional profiles, mapped into 8 emotion clusters. It aims to predict cluster membership for new images and identify the visual features driving emotional responses.

Possible challenges:

1. Image classification: train a deep neural network to predict cluster membership from images
2. Feature extraction: estimate feature contributions via controlled datasets and cluster distributions
3. Recommender system: suggest image styles based on (synthetic) user preferences

### Task Type

Image Classification / Feature Extraction / Recommender System

### Results Summary

#### Best Model Performance

- **Best Model:** ResNet50 transfer-learning classifier with ImageNet normalization
- **Evaluation Metric:** Weighted F1-score, supported by accuracy, weighted precision, weighted recall, macro F1, loss, per-class metrics, and a confusion matrix
- **Final Performance:** On the held-out test split of 1,088 images, the ResNet50 model achieved 1.3423 test loss, 54.60% accuracy, 0.5650 weighted precision, 0.5460 weighted recall, 0.5524 weighted F1, and 0.5107 macro F1.

#### Model Comparison

- **Baseline Performance:** The initial custom CNN baseline achieved 39.98% accuracy, 0.3748 weighted F1, 0.3883 weighted precision, 0.3998 weighted recall, and 2.9650 loss on the test split.
- **Improvement Over Baseline:** ResNet50 improved accuracy by about 14.6 percentage points and weighted F1 by about 17.8 percentage points compared with the custom CNN baseline on the held-out test split.
- **Best Alternative Model:** AlexNet transfer learning was the strongest alternative saved experiment, reaching 49.91% accuracy, 0.4816 weighted F1, and 0.4407 macro F1 on the held-out test split.

#### Key Insights

- **Most Important Features:** The project does not compute explicit feature-importance scores. Performance is mainly driven by learned visual representations from the ImageNet-pretrained ResNet50 backbone, ImageNet-compatible normalization, RGB image content, and training augmentation through horizontal flips and small rotations.
- **Model Strengths:** The model performs best on `intellectual_unease`, `lighthearted_humor`, and `sublime_activation`, with final test F1-scores of 0.66, 0.66, and 0.61 respectively in the notebook classification report.
- **Model Limitations:** `energetic_playfulness` is the weakest class, with a final test F1-score of 0.27. Several emotion clusters are visually overlapping and subjective, and the dataset is imbalanced, so errors remain likely for subtle or ambiguous aesthetic responses.
- **Business Impact:** The model is strong enough to support exploratory affective image tagging, dataset triage, and prototype recommendation or content-analysis workflows, but its current accuracy is not sufficient for high-stakes or fully automated decisions without human review.

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation.ipynb)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)

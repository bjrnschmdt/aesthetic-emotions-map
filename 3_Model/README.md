# Model Definition and Evaluation

**Notebook:** [model_definition_evaluation.ipynb](model_definition_evaluation.ipynb)

## Overview

This milestone defines and evaluates the final image-classification model for the
Aesthetic Emotions Map project. The task is to classify images into eight
aesthetic-emotion labels:

- `aesthetic_emptiness`
- `energetic_playfulness`
- `intellectual_unease`
- `lighthearted_humor`
- `melancholic`
- `pure_calm`
- `serene_beauty`
- `sublime_activation`

The final workflow uses the Hugging Face dataset
`bjoern-doege/aesthetic-emotions-map`, combines the provided train and validation
splits for cross-validation/training, and keeps the provided test split untouched
for final evaluation.

## Model Selection

Several image-classification approaches were considered during the modelling
work:

- A custom CNN baseline from the previous milestone.
- A custom CNN with grouped k-fold validation.
- An Optuna-tuned CNN experiment.
- AlexNet transfer learning.
- ResNet50 transfer learning.
- An HSV preprocessing variant.

The selected final model is **ResNet50 transfer learning with ImageNet
normalization**. This model was chosen because it produced the strongest
weighted F1 score among the evaluated approaches. Transfer learning is also a
good fit for this dataset because the model can reuse general visual features
learned from ImageNet while only adapting the final classification head to the
eight aesthetic-emotion classes.

## Feature Engineering

The project is an image-classification task, so feature engineering is handled
through image preprocessing and augmentation rather than tabular feature
construction.

Training images are transformed with:

- RGB conversion.
- Random horizontal flipping with probability `0.5`.
- Random rotation up to `10` degrees.
- Tensor conversion.
- ImageNet normalization with mean `[0.485, 0.456, 0.406]` and standard
  deviation `[0.229, 0.224, 0.225]`.

Validation and test images use deterministic preprocessing:

- RGB conversion.
- Tensor conversion.
- The same ImageNet normalization.

The notebook also preserves the dataset's `style` field as a grouping variable.
This prevents visually or stylistically related samples from being split across
training and validation folds.

## Hyperparameter Tuning

The final workflow uses grouped k-fold cross validation rather than a simple random
split. The notebook applies `StratifiedGroupKFold` with:

- `n_splits = 5`
- `random_state = 42`
- stratification by emotion label
- grouping by image `style`

This setup keeps class proportions balanced while reducing leakage between
stylistically similar images. The main training configuration is:

- batch size: `64`
- learning rate: `0.001`
- epochs: `10`
- optimizer: Adam
- loss: weighted cross-entropy

Weighted cross-entropy is used to reduce the effect of class imbalance. The
notebook loads saved fold metrics and checkpoints from `3_Model/experiments/`
when available; otherwise it trains the folds and writes the resulting metrics.

## Implementation

The final model is implemented in PyTorch with `torchvision.models.resnet50`
using `ResNet50_Weights.DEFAULT`.

Implementation details:

- The pretrained ResNet50 feature layers are frozen.
- The final fully connected layer is replaced with a new linear layer whose
  output size equals the eight emotion classes.
- Data loaders apply the training or evaluation transforms lazily through the
  Hugging Face dataset transform API.
- Training and evaluation are seeded for reproducibility across Python, NumPy,
  and PyTorch.
- The code selects CUDA, Apple MPS, or CPU depending on local hardware
  availability.

The final checkpoint used by the notebook is:

`3_Model/experiments/model_seed_42_w_k-fold_transfer_learning_resnet50_imagenet_normalized.pth`

## Evaluation Metrics

The notebook evaluates the classifier with:

- accuracy
- weighted precision
- weighted recall
- weighted F1
- macro F1
- cross-entropy loss
- per-class precision, recall, F1, and support
- confusion matrix
- rough human survey baseline metrics from
  `aem_human_baseline_2026_06_25_a.csv`

Accuracy gives a simple overall correctness score. Weighted F1 is the primary
comparison metric because the class distribution is not uniform, and it reflects
performance across all samples while still accounting for precision/recall
tradeoffs. Macro F1 is also reported to show how evenly the model performs
across classes independent of class frequency.

The human baseline is included as contextual evidence for task difficulty. It
comes from a small survey with `N=5` participants who assigned a stratified
subset of `32` images to the same emotion clusters. The subset contains `4`
images per emotion cluster: `2` images with the `antibiotics` motif and `2`
images with the `landscape` motif for each of the eight clusters. Because this
survey subset is much smaller than the full held-out test split, these values
should not be interpreted as a direct test-set benchmark.

### Human Survey Baseline

| Metric    | Mean   | Std. deviation | Median |
| --------- | -----: | -------------: | -----: |
| Accuracy  | 0.3375 |         0.0559 | 0.3750 |
| Precision | 0.3431 |         0.0569 | 0.3241 |
| Recall    | 0.3375 |         0.0559 | 0.3750 |
| F-measure | 0.3868 |         0.0481 | 0.3898 |

### Final Test Results

The notebook evaluates the trained ResNet50 model on the untouched test split of
`1,088` images.

| Metric             |  Score |
| ------------------ | -----: |
| Test loss          | 1.3423 |
| Accuracy           | 0.5460 |
| Weighted precision | 0.5650 |
| Weighted recall    | 0.5460 |
| Weighted F1        | 0.5524 |
| Macro F1           | 0.5107 |

The strongest per-class results are for `lighthearted_humor`,
`intellectual_unease`, and `sublime_activation`. The weakest class is
`energetic_playfulness`, which has lower precision and F1 than the other
classes.

## Comparative Analysis

The final ResNet50 transfer-learning model improves over the baseline and the
other saved experiments.

| Model                      | Evaluation scope                         | Accuracy | F1     | Macro F1 | Notes                                             |
| -------------------------- | ---------------------------------------- | -------: | -----: | -------: | ------------------------------------------------- |
| ResNet50 transfer learning | Full held-out test split                 |   0.5460 | 0.5524 |   0.5107 | Selected final model with ImageNet normalization; F1 is weighted F1. |
| AlexNet transfer learning  | Full held-out test split                 |   0.4991 | 0.4816 |   0.4407 | Transfer-learning alternative; F1 is weighted F1. |
| Custom CNN baseline        | Full held-out test split                 |   0.3998 | 0.3748 |      N/A | Initial baseline from `2_BaselineModel/baseline_model.ipynb`; F1 is weighted F1. |
| Custom CNN + k-fold        | Full held-out test split                 |   0.4191 | 0.4118 |   0.3661 | Grouped k-fold custom CNN experiment; F1 is weighted F1. |
| Human survey baseline      | 32-image subset; 4 images per cluster, split evenly across `antibiotics` and `landscape`; `N=5` humans |   0.3375 | 0.3868 |      N/A | Rough human context only; subset differs from the model test split. |

Compared with the original custom CNN baseline, ResNet50 increases accuracy by
about `14.6` percentage points and weighted F1 by about `17.8` percentage points.
The improvement likely comes from pretrained visual representations and the use
of ImageNet-compatible normalization. The remaining errors show that several
aesthetic-emotion classes are visually overlapping and subjective, so further
improvements may require more data, stronger fine-tuning, label-quality review,
or model architectures that can capture higher-level semantic context.

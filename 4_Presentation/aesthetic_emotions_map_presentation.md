# Aesthetic Emotions Map

	Visual emotion classification from AI-generated image styles

	Project presentation

	June 25, 2026

Opening: this project asks whether image pixels alone can predict the aesthetic emotion cluster assigned to an AI-generated style.

---

# Roadmap

	- Problem and related work
	- Dataset and split strategy
	- Human baseline survey
	- Baseline CNN
	- Style-aware k-fold validation
	- Optuna hyperparameter tuning
	- Transfer learning with MobileNetV3
	- Results, challenges, and next steps

The story follows the modeling path: start simple, make evaluation more reliable, tune the architecture, then use pretrained visual features.

---

# Problem and Context

	- Task: multi-class image classification
	- Input: RGB images, 224 x 224 pixels
	- Output: one of 8 aesthetic emotion clusters
	- Goal: predict cluster membership for unseen visual styles
	- Motivation: connect visual style with expected emotional response

This is not object recognition. The target is a synthetic aesthetic-emotion label, so the model needs to learn patterns connected to mood, intensity, calmness, humor, unease, and beauty.

---

# Related Work

	- Image emotion recognition benchmarks show CNNs outperform handcrafted affective features.
	- ArtEmis connects artworks with emotion annotations and explanations.
	- Emotion-specific pretraining can improve transfer to affective vision tasks.

The literature supports three decisions in this project: use CNNs as the core model family, evaluate carefully because emotion labels are subjective, and test transfer learning instead of relying only on a small custom CNN.

---

# Dataset and Split

	- Source: Hugging Face dataset `bjoern-doege/aesthetic-emotions-map`
	- Size: 6,832 AI-generated images
	- Structure: 427 styles x 4 motifs x 4 variations
	- Split: train 70%, validation 15%, test 15%
	- Test set: 1,088 images
	- Split unit: style, not individual image

The split was done at style level. This matters because images from the same style are visually related; random image-level splitting would make the test set too easy.

---

# Emotion Clusters

	- Sublime Activation
	- Intellectual Unease
	- Energetic Playfulness
	- Lighthearted Humor
	- Aesthetic Emptiness
	- Melancholic
	- Pure Calm
	- Serene Beauty

The classes are imbalanced. Accuracy is useful as a headline metric, but weighted F1 and macro F1 are needed to understand whether minority emotion clusters are being handled.

---

# Preprocessing and Metrics

	- RGB images resized to model input resolution
	- Tensor conversion and channel normalization
	- Dataset mean: `[0.5111, 0.4883, 0.4648]`
	- Dataset std: `[0.3434, 0.3208, 0.3226]`
	- Metrics: accuracy, weighted precision, weighted recall, weighted F1, macro F1

The baseline uses raw pixels. The transfer-learning model uses the preprocessing expected by the pretrained MobileNetV3 weights.

---

# Human Baseline

	- Survey participants: N=5
	- Task: humans assigned images to the same emotion clusters

	| Metric | Mean | Std. deviation | Median |
	|:--|--:|--:|--:|
	| Accuracy | 0.3375 | 0.0559 | 0.3750 |
	| Precision | 0.3431 | 0.0569 | 0.3241 |
	| Recall | 0.3375 | 0.0559 | 0.3750 |
	| F-measure | 0.3868 | 0.0481 | 0.3898 |

The human baseline gives a useful reality check. Even people had difficulty mapping images to these eight abstract emotion clusters, which supports treating the task as subjective and non-trivial.

---

# Baseline CNN

	- Custom CNN trained from scratch
	- Three convolution + pooling blocks
	- Fully connected classifier with dropout
	- Optimizer: Adam
	- Training: 10 epochs

	| Test metric | Score |
	|:--|--:|
	| Accuracy | 0.3998 |
	| Weighted precision | 0.3883 |
	| Weighted recall | 0.3998 |
	| Weighted F1 | 0.3748 |
	| Loss | 2.9650 |

The baseline is intentionally simple: enough capacity to learn spatial features, but not a large pretrained architecture. Note: `2_BaselineModel/README.md` reports 47.15% accuracy, but the executed notebook output in `baseline_model.ipynb` reports 39.98%. This deck uses the executed notebook output for consistency.

---

# Style-Aware K-Fold

	- Train + validation combined for cross-validation: 5,744 images
	- Test set kept untouched: 1,088 images
	- Splitter: `StratifiedGroupKFold`
	- Groups: image style
	- Overlap between train+validation styles and test styles: 0

The important change is evaluation design. Stratification keeps label distribution more stable; grouping prevents related styles from crossing between folds.

---

# K-Fold CNN Result

	| Evaluation | Metric | Score |
	|:--|:--|--:|
	| 5-fold CV | Accuracy | 0.3486 +/- 0.0428 |
	| 5-fold CV | Macro F1 | 0.3045 +/- 0.0410 |
	| Final test | Accuracy | 0.4237 |
	| Final test | Weighted F1 | 0.4050 |
	| Final test | Macro F1 | 0.3659 |

The final CNN trained on train+validation improves over the first baseline on the test set, but the cross-validation variance shows that generalizing to unseen styles is unstable.

---

# Optuna Tuning

	- Search method: Optuna
	- Objective: mean CV macro F1
	- Best CV macro F1: 0.3197
	- Best layers: 3
	- Filters: 40, 40, 64
	- Learning rate: 0.000517
	- Resolution: 64
	- Batch size: 8

Optuna improved the validation objective compared with the untuned CNN cross-validation macro F1, but this did not transfer to the held-out test set.

---

# Optuna Test Result

	| Test metric | Score |
	|:--|--:|
	| Accuracy | 0.3493 |
	| Weighted precision | 0.4249 |
	| Weighted recall | 0.3493 |
	| Weighted F1 | 0.3593 |
	| Macro F1 | 0.3206 |

The tuned model underperformed the k-fold CNN on the test set. One likely factor is that the selected 64 x 64 input resolution may have removed visual detail needed for emotion classification.

---

# Transfer Learning

	- Model: MobileNetV3 Small
	- Initialization: pretrained ImageNet weights
	- Classifier head replaced with 8 output classes
	- Training: style-aware 5-fold workflow
	- Final model trained on train + validation, evaluated once on test

Transfer learning changes the starting point. Instead of learning all visual features from 6,832 images, the model reuses generic visual features learned from a much larger image dataset.

---

# Implementation Excerpt

```python
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def create_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 8)
    return model
```

This is the core transfer-learning change: keep the pretrained feature extractor and replace the last classifier layer so the network predicts the eight emotion clusters.

---

# Transfer Result

	| Evaluation | Metric | Score |
	|:--|:--|--:|
	| 5-fold CV | Accuracy | 0.4576 +/- 0.0251 |
	| 5-fold CV | Macro F1 | 0.4167 +/- 0.0334 |
	| Final test | Accuracy | 0.5092 |
	| Final test | Weighted F1 | 0.4998 |
	| Final test | Macro F1 | 0.4516 |

This was the best model in the experiments. It improved both headline accuracy and macro F1, which is important because macro F1 gives each class equal weight.

---

# Model Comparison

	| Model / baseline | CV macro F1 | Accuracy | F1 / F-measure | Macro F1 |
	|:--|--:|--:|--:|--:|
	| Human survey baseline, N=5 | n/a | 0.3375 | 0.3868 | n/a |
	| Baseline CNN | n/a | 0.3998 | 0.3748 | n/a |
	| CNN + style-aware k-fold | 0.3045 | 0.4237 | 0.4050 | 0.3659 |
	| Optuna-tuned CNN | 0.3197 | 0.3493 | 0.3593 | 0.3206 |
	| MobileNetV3 transfer learning | 0.4167 | 0.5092 | 0.4998 | 0.4516 |

The strongest result comes from transfer learning. Compared with the human survey mean, test accuracy increased by about 17.2 percentage points. Compared with the executed baseline CNN notebook, it increased by about 10.9 percentage points.

---

# Interpretation

	- Style-aware evaluation made the task more realistic.
	- The human baseline confirms that the label mapping is difficult.
	- The custom CNN learned the training data but struggled to generalize.
	- Optuna improved the search objective but not the final holdout result.
	- Pretrained visual features generalized better to unseen styles.

The pattern suggests that feature representation is the main bottleneck. The best model outperformed the small human survey group, but the survey also shows that the target itself is not an obvious visual categorization task.

---

# Challenges

	- Preventing style leakage between splits
	- Handling imbalanced emotion clusters
	- Comparing model metrics with a small human survey baseline
	- Avoiding overfitting on a small image dataset
	- Reconciling validation improvements with final test performance
	- Keeping the final test set untouched until model selection was complete

The main challenge was evaluation validity. Because each style has multiple related images, leakage would have made the model look better than it really is.

---

# Limitations and Future Work

	- Labels are synthetic emotion clusters, not direct human ratings.
	- Dataset size is modest for training CNNs from scratch.
	- Some emotion classes have fewer images than others.
	- Analyze class-level errors with confusion matrices.
	- Try stronger pretrained architectures and augmentation.
	- Use explainability methods such as Grad-CAM.

The model is useful as a first classifier, but not yet reliable enough for high-stakes or personalized emotion prediction.

---

# Conclusion

	- Best model: MobileNetV3 transfer learning
	- Human baseline accuracy, N=5: 0.3375
	- Final test accuracy: 0.5092
	- Final weighted F1: 0.4998
	- Final macro F1: 0.4516
	- Main lesson: pretrained visual features outperform small CNNs trained from scratch

The project shows that aesthetic emotion clusters are learnable from images, but robust generalization to unseen styles remains difficult.

---

# Sources

	- Project notebooks and README files in this repository
	- Human baseline metrics: `3_Model/aem_human_baseline_2026_06_25_a.csv`
	- iA Presenter Markdown Guide: https://ia.net/presenter/support/basics/markdown
	- You et al., image emotion recognition benchmark
	- Achlioptas et al., ArtEmis
	- Wei et al., visual emotion representations from web data

These sources support the presentation structure, markdown format, and modeling rationale.

---

# Q&A

	Questions?

# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection

- **Baseline Model Type:** Convolutional Neural Network (CNN)
- **Rationale:** CNNs are effective for image classification tasks due to their ability to capture spatial hierarchies and learn relevant features directly from the raw pixel data. This makes them a suitable choice for our aesthetic emotions map project, where the goal is to classify images based on the emotions they evoke.

### Model Performance

- **Evaluation Metric:** Accuracy, F1-Score, Precision, Recall
- **Performance Score:** 47.15% accuracy, F1-score of 0.4593, precision of 0.4639, and recall of 0.4715 on the test set.
- **Cross-Validation Score:** Did not perform cross-validation for the baseline model, because splits were done manually.

### Evaluation Methodology

- **Data Split:** Manual split with Train/Validation/Test split ratios, e.g., 70/15/15
- **Evaluation Metrics:** Accuracy, because it provides a straightforward measure of overall performance. F1-score, precision, and recall were also used to provide a more comprehensive evaluation of the model's performance, especially because of class imbalances in the dataset.

### Metric Practical Relevance

[Explain the practical relevance and business impact of each chosen evaluation metric. How do these metrics translate to real-world performance and decision-making? What do the metric values mean in the context of your specific problem domain?]

- **Accuracy:** Indicates the overall correctness of the model's predictions. In the context of our project, a higher accuracy means that the model is better at correctly classifying images into their respective emotion categories, which is crucial for applications like affective image content analysis (AICA).
- **F1-Score:** Provides a balance between precision and recall, which is important in cases where there may be class imbalances. A higher F1-score indicates that the model is performing well in terms of both precision and recall, which is essential for ensuring that the model is not only accurate but also reliable in identifying true positives and minimizing false positives.
- **Precision:** In our context, a high precision means that when the model predicts a certain emotion, it is likely to be correct, which is important for applications that require high confidence in the predictions, such as predicting user emotions for visualizations on mental health.
- **Recall:** A high recall means that the model is effectively identifying most of the relevant instances, which is prefered in applications where it is beneficial to capture as many relevant instances as possible, such as in recommendation systems.

## Next Steps

This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.

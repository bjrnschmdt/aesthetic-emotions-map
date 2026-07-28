	###### 02.07.2026 Machine Learning Degree Presentation
	# Aesthetic Emotion Classification
	Björn Döge, Dmitry Vergeles, Cem Çakmak

---
## Introduction

Our project aims to predict the emotions evoked by visual styles in images using machine learning. 
But before we look at training a model let's see if we can train you for the task.
---
	**Intellectual Unease** 
	Feels like walking through a maze that is intriguing but slightly unsettling.

/assets/landscape_--sref_543782035_--v_7_a525e90b-aae7-4b40-b80b-0f1e45c127f6_3.png

/assets/landscape_--sref_1085397630_--v_7_a10bbd32-dac8-4d41-86c7-31797aaef44d_3.png

/assets/landscape_--sref_1155621117_--v_7_64edba22-5990-4b3e-820e-d87a7d6e80cd_3.png

/assets/landscape_--sref_1236556796_--v_7_ae9639e7-3762-4a60-b9e9-d6999eb50ded_3.png

/assets/landscape_--sref_2950498809_--v_7_7bf00317-4d4b-45d1-a2c1-9e157e5ae6da_3.png

/assets/landscape_--sref_3146651154_--v_7_432c7010-2325-474f-8022-29588f98faf2_3.png

/assets/landscape_--sref_3549065407_--v_7_fe4a71d8-20e1-42ed-b879-e338f72c1d64_3.png

Here you see samples from the "Intellectual Unease" class.
The styles should feel like "walking through a maze that is intriguing but slightly unsettling". 
Do you feel the uneasiness?

---
	**Melancholic** 
	Feels like finding an old photograph and remembering a time that is gone forever.

/assets/landscape_--sref_1780687264_--v_7_1b7ba7f9-53bc-465c-ab57-8a2c41538423_3.png

/assets/landscape_--sref_3996291997_--v_7_a84dd27c-8ec9-46f3-acf8-69c3b0828c35_3.png

/assets/landscape_--sref_37284191_--v_7_80cb57e4-3e15-4fef-9258-53fa3c014a74_3.png

Here are samples from the "Melancholic" class.
The styles should feel like "finding an old photograph and remembering a time that is gone forever". 
Do you feel the sadness?

---
	**Energetic Playfulness** 
	Feels like dancing when your favorite song unexpectedly comes on.

/assets/landscape_--sref_1017203228_--v_7_2c81040d-9308-458e-93cf-b71ce5e8acf0_3.png

/assets/landscape_--sref_1189783625_--v_7_5eea9f10-0054-484e-93f9-e35ee1048eaf_3.png

These are samples from the "Energetic Playfulness" class.
The styles should feel like "dancing when your favorite song unexpectedly comes on". 
Do you feel energized?

---
	Can we train a machine to predict emotion categories solely based on visual features?

/assets/landscape_--sref_3562468608_--v_7_a48f3d2e-0356-4ce3-a829-726e201c160a_3.png

Let's put you to the test.
Which class would you assign this image to? 
Intellectual Unease, Melancholic or Energetic Playfulness?
Not easy, right? According to our dataset it is "Energetic Playfulness".
We chose this ambiguous style on purpose to show how difficult and somewhat subjective the task is.
So the question remains, can we train a machine to predict emotion categories solely based on visual features?
---
## Literature Review

---
	**Large-Scale Image Emotion Recognition:** Fine-tuned CNNs trained on strongly labeled emotion images significantly outperform handcrafted features and generic ImageNet models across eight emotion categories.[^https://cdn.aaai.org/ojs/9987/9987-13-13515-1-2-20201228.pdf]

	**ArtEmis:** Links 80,000 artworks with 450,000 emotional reactions and explanations, showing that subjective responses to art still contain learnable patterns.[^https://openaccess.thecvf.com/content/CVPR2021/papers/Achlioptas_ArtEmis_Affective_Language_for_Visual_Art_CVPR_2021_paper.pdf]

	**Learning from Web Data:** EmotionNet uses over one million emotion-tagged stock images to learn affective visual features that generalize better than standard ImageNet features.[^https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Learning_Visual_Emotion_Representations_From_Web_Data_CVPR_2020_paper.pdf]  

The literature supports three decisions in this project: use CNNs as the core model family, evaluate carefully because emotion labels are subjective, and test transfer learning instead of relying only on a small custom CNN.

---
## Dataset Characteristics
---
	`6,832` AI-generated RGB images at `224 × 224` pixels
	`427` styles × `4` motifs × `4` variations
	Labels: `8` Emotion Categories
	Style-level split train/validation/test (70/15/15)
	Labels derived from: cluster analysis of LLM emotion annotations

In a prior study we generated 8000 images using midjourney by only using the motif + style code e.g. `bacteria --sref 4255112626`
We then let an LLM annotate the images using the AESTHEMOS questionnaire to get an emotion profile per image
After that we performed cluster analysis on those 21 dimensional profiles to get 8 clusters that form our labels 
---
/assets/serene_beauty_grid.png
size: contain

Here you can see all sixteen images for one style
The Rows indicate the motifs while the columns indicate the variation
---
|                       | Train Set | Validation Set | Test Set |
| :-------------------- | :-------- | :------------- | :------- |
| Intellectual Unease   | 23%       | 23.4%          | 22.0%    |
| Sublime Activation    | 19%       | 18.8%          | 19.1%    |
| Lighthearted Humor    | 13.2%     | 14.0%          | 13.2%    |
| Serene Beauty         | 11.9%     | 10.9%          | 11.8%    |
| Pure Calm             | 11.2%     | 10.9%          | 11.8%    |
| Melancholic           | 8.5%      | 9.4%           | 8.8%     |
| Aesthetic Emptiness   | 7.8%      | 7.8%           | 7.4%     |
| Energetic Playfulness | 5.4%      | 4.7%           | 5.9%     |
Chart: Bar
Orientation: horizontal
xLabel: Labels
yLabel: Distribution of labels
Caption: Bar Chart illustrating the label imbalance

The dataset is highly imbalanced: most images belong to the `intellectual unease` class while the least images belong to the `energetic playfulness` class.
We tried to retain the relativ distribution across train set, validation set and test set. 
---
## Baseline Model
---

/assets/baseline_cnn_dark 1.png
size: contain

Our baseline is a simple CNN trained from scratch. 
It has three convolutional layers with ReLU activation and max-pooling, using 32, 64, and 128 feature maps. 
We use three-by-three kernels throughout the network, followed by a fully connected classifier with dropout. 
The model was trained for ten epochs using the Adam optimizer.

---
## Baseline Model
| Metric             |  Score |
| :----------------- | -----: |
| Accuracy           | 0.3998 |
| Weighted precision | 0.3883 |
| Weighted recall    | 0.3998 |
| Weighted F1        | 0.3748 |
| Loss               | 2.9650 |

Although the model achieved high training accuracy, it failed to generalize to unseen data, indicating overfitting.
Still, it performed better than random guessing, which would achieve an accuracy of 12.5%.

---
## Model Definition and Evaluation
---

/assets/Bildschirmfoto 2026-07-01 um 15.39.45.png
size: contain

To put the model’s performance into perspective, we estimated human performance on the same task.
We created a survey with 32 images—four per emotion cluster across two motifs—and asked five participants to classify them.

---
### Human Baseline Estimation (N = 5)
 
| Metric    |   Mean | Std. deviation | Median |
| :-------- | -----: | -------------: | -----: |
| Accuracy  | 0.3375 |         0.0559 | 0.3750 |
| Precision | 0.3431 |         0.0569 | 0.3241 |
| Recall    | 0.3375 |         0.0559 | 0.3750 |
| F-Score   | 0.3868 |         0.0481 | 0.3898 |

Even humans struggled to classify the eight abstract emotion clusters, showing that this is a subjective and challenging task.
The human survey provides context for task difficulty, but it is not a direct full-test-set benchmark.

---
### Hyperparameter Optimization with k-fold Cross Validation

We then optimized the baseline model using Optuna with five-fold cross-validation.
We also replaced the standard loss with a weighted loss function to compensate for class imbalance.
Despite tuning the hyperparameters, performance was slightly worse than the original baseline.

| Item             | Value                            |
| :--------------- | :------------------------------- |
| Method           | Optuna                           |
| Objective        | Maximize mean weighted F1-score  |
| Evaluation       | 5-fold cross-validation          |
| Metric           | Average weighted F1 across folds |
| Number of trials | 20                               |

| Hyperparameter      | Search space    | Best value |
| :------------------ | :-------------- | :--------- |
| n layers            | 1–3             | 3          |
| n filters layer 0   | 8–64, step 8    | 48         |
| n filters layer 1   | 8–64, step 8    | 32         |
| n filters layer 2   | 8–64, step 8    | 48         |
| kernel size layer 5 | [3, 5]          | 5          |
| kernel size layer 5 | [3, 5]          | 5          |
| kernel size layer 5 | [3, 5]          | 5          |
| Dropout             | 0.1–0.5         | 0.42       |
| fc size             | 64–512, step 64 | 128        |
| Learning rate       | 1e-4–1e-2       | 1e-3       |
| Dropout             | 0.1–0.5         | 0.4        |
| Resolution          | [16, 32, 64]    | 64         |
| Batch Size          | [8, 16, 32, 64] | 16         |

| Tuning metric           | score |
| :---------------------- | :---- |
| Weighted F1 (5-fold CV) | 0.32  |

---
// ### Transfer Learning using ResNet50

```python
def update_model_last_layer(model, num_classes):
    """
    Freezes the feature layers of a pre-trained model and replaces its final
    classification layer with a new one adapted to the specified number of classes.

    Args:
        model (torch.nn.Module): The pre-trained model to be modified.
        num_classes (int): The number of output classes for the new classification layer.

    Returns:
        torch.nn.Module: The modified model with frozen feature layers and a new
                         classification layer.
    """


    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    last_classifier_layer = model.fc 
    
    num_features = last_classifier_layer.in_features
    
    new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)
    
    # Replace the original last classification layer with the newly created layer
    model.fc = new_classifier

    return model
```

```python
def create_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = update_model_last_layer(model, num_classes=8)
    return model.to(DEVICE)
```

We therefore turned to transfer learning.
We selected ResNet50, a widely used model pretrained on over one million ImageNet images.
We froze the pretrained feature extractor and replaced the final classification layer to predict our eight emotion clusters.
---
## Results
---
|                             | Accuracy | Precision | Recall | F1-Score |
| :-------------------------- | :------- | :-------- | :----- | :------- |
| Baseline Model              | 0.40     | 0.39      | 0.40   | 0.37     |
| Human Baseline              | 0.34     | 0.34      | 0.34   | 0.39     |
| Custom CNN + k-fold         | 0.42     | 0.42      | 0.42   | 0.41     |
| Transfer Learning ResNet 50 | 0.55     | 0.57      | 0.55   | 0.55     |
Chart: Bar
Orientation: horizontal
xLabel: Models
yLabel: Performance
Caption: Performance Metrics of various models compared to human baseline

As you can see, the transfer learning model clearly outperformed all previous approaches, achieving the best performance across every metric.

---
/assets/untitled-131.svg
size: contain

This confusion matrix shows the model’s predictions for each class.
The y-axis represents the true labels, and the x-axis the predicted labels. A strong diagonal indicates good performance.
The model classified Intellectual Unease reliably but struggled with Energetic Playfulness.
It frequently confused Energetic Playfulness with Lighthearted Humor, as well as Melancholic with Intellectual Unease and Pure Calm with Serene Beauty.
---

/assets/bacteria_--sref_4255112626_--v_7_ceae2929-0637-45a1-a521-3e455ceed8bb_0.png
caption: True: Energetic Playfulness</br>Pred: Energetic Playfulness 

/assets/ear_infection_--sref_1155621117_--v_7_4ddf6ba6-6aea-4a12-b908-d00d9c585e2a_3.png
caption: True: Intellectual Unease</br>Pred: Sublime Activation

/assets/landscape_--sref_2414593783_--v_7_d3e7f12d-707f-4c48-a8d8-f0590e43e3a5_1.png
caption: True: Pure Calm</br>Pred: Lighthearted Humor

/assets/ear_infection_--sref_93165945_--v_7_df2154b1-e169-4fcc-b083-94b368c31424_3.png
caption: True: Sublime Activation</br>Pred: Sublime Activation 

/assets/antibiotics_--sref_658225328_--v_7_e6e4c92e-0523-4ab4-9d64-b8afabfd2c16_1.png
caption: True: Aesthetic Emptiness</br>Pred: Aesthetic Emptiness

/assets/bacteria_--sref_1386943109_--v_7_764c2678-9a6a-4a4a-b2eb-08de3b2b9a9d_0.png
caption: True: Intellectual Unease</br>Pred: Intellectual Unease

/assets/ear_infection_--sref_1236556796_--v_7_aee16f54-6172-45cf-8fff-02a603c73cce_2.png
caption: True: Intellectual Unease</br>Pred: Intellectual Unease

/assets/bacteria_--sref_382149816_--v_7_592bc712-09ad-4c22-be13-1913fe60faa0_3.png
caption: True: Pure Calm</br>Pred: True Calm

/assets/ear_infection_--sref_2002690750_--v_7_97853504-e0b3-45b4-88b6-dd2b19cd6ccf_0.png
caption: True: Sublime Activation</br>Pred: Sublime Activation

/assets/bacteria_--sref_3965783643_--v_7_e3d33792-ed0c-4c6e-aa1d-b2ae3381e504_2.png
caption: True: Lighthearted Humor</br>Pred: Energetic Playfulness

/assets/landscape_--sref_3768064499_--v_7_5440986f-29d6-424f-85e5-47af0e575995_1.png
caption: True: Intellectual Unease</br>Pred: Intellectual Unease

/assets/bacteria_--sref_1757852444_--v_7_87469a35-f9c3-4f5c-be81-8c82f346a05a_0.png
caption: True: Lighthearted Humor</br>Pred: Lighthearted Humor

Here are twelve randomly selected test images with their ground-truth labels and the model’s predictions.
The model classified eight correctly and misclassified four.
With that, I’ll hand over to Cem.

---
## Challenges and Errors
---
	- Heterogeneous visual styles within classes
	- Gradual and overlapping class boundaries
	- LLM-generated annotations lack human validation
	- Potential mismatch between initial and articulated human emotions
	- Preventing style leakage across data splits
	- Handling imbalanced emotion classes
	- Comparing model performance against a small human baseline
	- Avoiding overfitting on a limited image dataset

---
## Discussion
---
	- More training data could improve performance
	- Emotion labels are inherently subjective
	- Training may be sensitive to random initialization (seed)


---
## Conclusion and Future Work

---
	* Model outperforms the human baseline (small sample)
	* Pretrained visual features outperform CNNs trained from scratch
	* Current model is not yet sufficiently reliable
	* Feature map analysis could provide additional insights
	* Personalized emotion prediction
---
## Q&A

# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and the Benchmark
  - **[Link](https://cdn.aaai.org/ojs/9987/9987-13-13515-1-2-20201228.pdf)**
  - **Objective**: Build a large, strongly labeled image-emotion dataset and test whether CNNs can outperform handcrafted visual features for multi-class visual emotion recognition.
  - **Methods**: The authors collected millions of weakly labeled emotion images, manually relabeled a subset with Amazon Mechanical Turk, and trained CNN models for eight emotion categories: amusement, awe, contentment, excitement, anger, disgust, fear, and sadness. They compare ImageNet features, noisy fine-tuning, and emotion-specific fine-tuning.
  - **Outcomes**: Fine-tuned CNNs substantially improved over generic ImageNet-CNN features and handcrafted affective features. The paper establishes CNN-based emotion classification as a valid benchmark task.
  - **Relation to the Project**: The paper provides a useful reference for dataset setup, CNN training, and evaluation strategies for emotion-related image classification.

- **Source 2**: ArtEmis: Affective Language for Visual Art
  - **[Link](https://openaccess.thecvf.com/content/CVPR2021/papers/Achlioptas_ArtEmis_Affective_Language_for_Visual_Art_CVPR_2021_paper.pdf)**
  - **Objective**: Create a large-scale dataset connecting artworks, emotional reactions, and natural-language explanations to study affective responses to visual art.
  - **Methods**: The authors collected over 450,000 emotion annotations and explanations for 80,000 artworks from WikiArt. Annotators selected one dominant emotion category and provided textual explanations. The paper trains CNN-based emotion classifiers and neural captioning models.
  - **Outcomes**: The study shows that emotional reactions to artworks are subjective but still contain meaningful agreement between annotators. CNNs were able to predict dominant emotional responses, and neural captioning systems generated plausible affective explanations.
  - **Relation to the Project**: The paper provides datasets, pre-trained models and notebooks in their repository, making it a valuable ressource for learning. The pretrained models could be used for transfer learning. Highlighting subjectivity the paper inspires me to think about personalized emotion prediction.

- **Source 3**: Learning Visual Emotion Representations from Web Data
  - **[Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Learning_Visual_Emotion_Representations_From_Web_Data_CVPR_2020_paper.pdf)**
  - **Objective**: Learn general-purpose visual emotion features from large-scale web data.
  - **Methods**: The authors build StockEmotion, a dataset with over one million stock images labeled with hundreds of emotion-related tags, and train EmotionNet as an emotion-specific feature extractor.
  - **Outcomes**: Emotion-specific pretraining improves performance across several visual emotion tasks and generalizes better than generic ImageNet features.
  - **Relation to the Project**: The paper is particularly relevant for the idea of extracting deep features that evoke emotional responses. In my project, I am also interested in analyzing the learned CNN features after training the classifier.

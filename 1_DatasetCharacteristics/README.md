# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source

- **Dataset Link:** [Provide a direct link to your dataset. If the dataset is private, explain the reason and provide contact information for the dataset owner]
- **Dataset Owner/Contact:** Björn Döge, bjoern.doege@icloud.com

### Dataset Characteristics

- **Number of Observations:** 6.832 ai-generated images (427 styles _ 4 motifs _ 4 variations)
- **Number of Features:** 3 color channels (RGB), 50.176 pixels per image (224x224 pixels)

### Target Variable/Label

- **Label Name:** Dataset consists only of an image folder and no csv file. There are folders for training, validation, and testing, each containing subfolders for the different classes (styles).
- **Label Type:** Classification
- **Label Description:** The label represents one of 8 aesthetic emotion categories.
- **Label Values:**
  - **Sublime Activation:** Broadly intense positive experience combining beauty, awe, enchantment, joy, energy, and fascination simultaneously. Cognitively stimulating as well as emotionally overwhelming. The richest aesthetic response in the dataset.
  - **Intellectual Unease:** Cognitively demanding and emotionally uncomfortable. Confusion, uneasiness, intellectual challenge, and surprise are elevated; joy and relaxation are strongly suppressed. Engaging but resistant to easy pleasure.
  - **Energetic Playfulness:** Lively and cheerful, defined by humor and joy alongside elevated energy and vitality. Actively fun and upbeat rather than quietly content. Negative emotions are absent.
  - **Lighthearted Humor:** Humor-dominant with elevated joy and relaxation. All cognitive and awe-related emotions are strongly suppressed. Amusing and pleasant but shallow — enjoyment without engagement.
  - **Aesthetic Emptiness:** Near-complete suppression of all positive emotions — awe, enchantment, beauty, joy, and fascination. Slight boredom elevation. No meaningful aesthetic response in any direction.
  - **Melancholic:** Defined by sadness, uneasiness, and nostalgia. Joy, beauty, enchantment, energy, and awe are strongly suppressed. Dark and heavy in tone — mournful rather than cognitively engaging.
  - **Pure Calm:** Relaxation as the sole signal, with no other emotions elevated. All activating, cognitive, and affective responses are suppressed. Restful and undisturbed, without aesthetic warmth or color.
  - **Serene Beauty:** Relaxation combined with awe, beauty, enchantment, being moved, and nostalgia. Calming yet emotionally resonant — quietly beautiful and gently moving without being activating.
- **Label Distribution:** Images are split into training (70%), validation (15%), and test (15%) sets with the following distribution:
  | Label | train | validation | test |
  |-----------------------|-------------|-------------|-------------|
  | Sublime Activation | 896 (19.0%) | 192 (18.8%) | 208 (19.1%) |
  | Intellectual Unease | 1088 (23%) | 240 (23.4) | 240 (22.0%) |
  | Energetic Playfulness | 256 (5.4%) | 48 (4.7%) | 64 (5.9%) |
  | Lighthearted Humor | 624 (13.2%) | 144 (14.0%) | 144 (13.2%) |
  | Aesthetic Emptiness | 368 (7.8%) | 80 (7.8%) | 80 (7.4%) |
  | Melancholic | 400 (8.5%) | 96 (9.4%) | 96 (8.8%) |
  | Pure Calm | 528 (11.2%) | 112 (10.9%) | 128 (11.8%) |
  | Serene Beauty | 560 (11.9%) | 112 (10.9%) | 128 (11.8%) |
  |-----------------------|-------------|-------------|-------------|
  | | 4720 | 1024 | 1088 |

### Feature Description

The features in the dataset are the pixel values of the images, which are represented as 3 color channels (RGB) with 50.176 pixels per image (224x224 pixels) and intensity values between 0 and 255. Image dimensions were chosen to conform to the ImageNet standard, which is commonly used in computer vision tasks. The mean and standard deviation of the pixel values across the dataset are as follows:

mean: [0.5111283659934998, 0.48830345273017883, 0.46479079127311707],
std: [0.3433663249015808, 0.3207928538322449, 0.32255250215530396]

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

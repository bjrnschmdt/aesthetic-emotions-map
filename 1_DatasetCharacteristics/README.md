# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source

- **Dataset Link:** [Link to the dataset on Huggingface](https://huggingface.co/datasets/bjoern-doege/aesthetic-emotions-map)
- **Dataset Owner/Contact:** Björn Döge, bjoern.doege@icloud.com

### Dataset Characteristics

- **Number of Observations:** 6.832 ai-generated images (427 styles _ 4 motifs _ 4 variations)
- **Number of Features:** 3 color channels (RGB), 50.176 pixels per image (224x224 pixels)

### Target Variable/Label

- **Label Name:** Dataset consists only of an image folder and no csv file. There are folders for training, validation, and testing, each containing subfolders for the different classes (emotion clusters).
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

<table>
  <tr>
    <td rowspan="2">
      Label
    </td>
    <td colspan="3">
      train (70%)
    </td>
    <td colspan="3">
      validation (15%)
    </td>
    <td colspan="3">
      test (15%)
    </td>
  </tr>
  <tr>
    <td>images</td>
    <td>styles</td>
    <td>percentage</td>
    <td>images</td>
    <td>styles</td>
    <td>percentage</td>
    <td>images</td>
    <td>styles</td>
    <td>percentage</td>
  </tr>

  <tr>
    <td>Sublime Activation</td>
    <td>896</td>
    <td>56</td>
    <td>19 %</td>
    <td>192</td>
    <td>12</td>
    <td>18.8 %</td>
    <td>208</td>
    <td>13</td>
    <td>19.1 %</td>
  </tr>

  <tr>
    <td>Intellectual Unease</td>
    <td>1088</td>
    <td>68</td>
    <td>23 %</td>
    <td>240</td>
    <td>15</td>
    <td>23.4 %</td>
    <td>240</td>
    <td>15</td>
    <td>22.0 %</td>
  </tr>

  <tr>
    <td>Energetic Playfulness</td>
    <td>256</td>
    <td>16</td>
    <td>5.4 %</td>
    <td>48</td>
    <td>3</td>
    <td>4.7 %</td>
    <td>64</td>
    <td>4</td>
    <td>5.9 %</td>
  </tr>

  <tr>
    <td>Lighthearted Humor</td>
    <td>624</td>
    <td>39</td>
    <td>13.2 %</td>
    <td>144</td>
    <td>9</td>
    <td>14.0 %</td>
    <td>144</td>
    <td>9</td>
    <td>13.2 %</td>
  </tr>

  <tr>
    <td>Aesthetic Emptiness</td>
    <td>368</td>
    <td>23</td>
    <td>7.8 %</td>
    <td>80</td>
    <td>5</td>
    <td>7.8 %</td>
    <td>80</td>
    <td>5</td>
    <td>7.4 %</td>
  </tr>

  <tr>
    <td>Melancholic</td>
    <td>400</td>
    <td>25</td>
    <td>8.5 %</td>
    <td>96</td>
    <td>6</td>
    <td>9.4 %</td>
    <td>96</td>
    <td>6</td>
    <td>8.8 %</td>
  </tr>

  <tr>
    <td>Pure Calm</td>
    <td>528</td>
    <td>33</td>
    <td>11.2 %</td>
    <td>112</td>
    <td>7</td>
    <td>10.9 %</td>
    <td>128</td>
    <td>8</td>
    <td>11.8 %</td>
  </tr>

  <tr>
    <td>Serene Beauty</td>
    <td>560</td>
    <td>35</td>
    <td>11.9 %</td>
    <td>112</td>
    <td>7</td>
    <td>10.9 %</td>
    <td>128</td>
    <td>8</td>
    <td>11.8 %</td>
  </tr>

  <tr>
    <td><strong>Total</strong></td>
    <td><strong>4720</strong></td>
    <td><strong>295</strong></td>
    <td><strong>100 %</strong></td>
    <td><strong>1024</strong></td>
    <td><strong>64</strong></td>
    <td><strong>100 %</strong></td>
    <td><strong>1088</strong></td>
    <td><strong>68</strong></td>
    <td><strong>100 %</strong></td>
  </tr>
</table>

The dataset was split at the style level rather than the image level to prevent data leakage between splits. Since each style is represented by multiple highly similar images across motifs and variations, random image-level splitting could place visually related samples in both training and evaluation sets. All images belonging to a given style were therefore assigned to a single split, enabling evaluation on previously unseen styles.

### Feature Description

The features in the dataset are the pixel values of the images, which are represented as 3 color channels (RGB) with 50.176 pixels per image (224x224 pixels) and intensity values between 0 and 255. Image dimensions were chosen to conform to the ImageNet standard, which is commonly used in computer vision tasks. The mean and standard deviation of the pixel values across the dataset are as follows:

```python
mean = [0.5111283659934998,
        0.48830345273017883,
        0.46479079127311707]
```

```python
std = [0.3433663249015808,
       0.3207928538322449,
       0.32255250215530396]
```

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

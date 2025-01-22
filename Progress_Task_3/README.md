# Progress Task 3

## Task Description

The objective of this task is to analyze and compare the narratives of disinformers and those of trustworthy sources within the dataset related to COVID-19. This involves identifying potential differences in linguistic patterns, themes, or other distinguishing features that characterize disinformation versus credible information.

To develop this task, we will use different NLP techniques to analyze the text data. Such as:

- Tokenization

- POS Tagging

- Named Entity Recognition (NER)

- Sentiment Analysis

- BERTopic for topic modeling

To visualize the results, we will use different graphical representations. There's a pdf file with the results of the analysis.

### Data

The data description can be found at the following link: [Fighting an Infodemic: COVID-19 Fake News Dataset](https://arxiv.org/pdf/2011.03327)

## Structure

```
📂 .    
    ├──── 📂 data
    ├──── 📄 Classificator.ipynb
    ├──── 📄 Preprocessing.ipynb
    ├────  ℹ️ README.md
    ├──── 📄 SentimentAnalysis.ipynb
    ├──── 📄 TopicModeling.ipynb
    └──── 📄 requirements.txt  
    
```

* 📂**data**: Contains the dataset used in the notebooks.
* **Clasiificator.ipynb**: Contains the code to classify the messages and analyze with shap, which are the most important features.
* **Preprocessing.ipynb**: Contains the exploratory data analysis and the preprocessing of the dataset.
* **SentimentAnalysis.ipynb**: Contains the code to analyze the sentiment of the messages.
* **TopicModeling.ipynb**: Contains the code to analyze the topics of the messages.

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

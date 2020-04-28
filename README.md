### Date created
2020-04-27

### Project Title
Disaster Response Pipeline

### Description
The aim of the project is to create a disaster response pipeline for messages in order to classify them and potentially take action to help people in emergency. The data is provided by Figure Eight and contains pre-labeled messages, and based on it a Machine Learning model with Grid Search is trained to optimize the classification into 36 categories. The user interface is a web app where future messages can be input and classified into those categories.

The pipeline consists of three parts:
1. ETL pipeline - extracts, transforms and loads the raw data of pre-labeled messages; two .csv files are merged, cleaned and stored in a structured database.
2. Machine Learning pipeline - applying Natural Language Processing (NLP) techniques on the text message data before training a multi-output Random Forest classifier with tuned hyper-parameters through Grid Search.

### Dependencies

- Python 3.7+
- Libraries: Pandas, numpy, sklearn, nltk, re, sqlalchemy, pickle, flask, plotly

### Installation

1. Clone this repo
2. Run the following commands in the root directory
- ETL pipeline command:
```python
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DataResponse.db
```
- ML pipeline command (this could take a while! Around 1-2 hours):
```python
python model/train_classifier.py data/DataResponse.db models/classifier.pkl
```
3. Run the web app by opening this URL in your browser: http://0.0.0.0:3001

### License
MIT

### Credits
Thanks to Udacity and Figure Eight for putting together this great project!

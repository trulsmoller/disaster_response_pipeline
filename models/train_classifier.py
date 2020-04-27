# import libraries
import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    '''
    This function loads data from a database into a DataFrame, then separates the categories part (X),
    the messages part (y) and the category labels (category_names) and returns them.
    
    Args:
    database_filepath (str)
    
    Returns:
    X (2d numpy array) - categories
    y (1d numpy array) - messages
    category_names (list of str)
    '''
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('df', engine)

    # extract category names
    df_y = df.iloc[:,4:]
    category_names = df_y.columns.tolist()

    # extract X and y
    df_X = df['message']
    X = df_X
    y = df_y
    
    return X, y, category_names


def tokenize(text):
    '''
    Args:
    text (str) - text from a text message
    
    Returns:
    clean_tokens (list of str) - list where each item is a word; tokenized and cleaned
    '''
    # replace urls with string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipe = Pipeline(steps = [
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # Initializing parameters for Grid search
    parameters = {}
    parameters['clf__estimator__n_estimators'] = [10, 50]    
    
    # GridSearch Object with pipeline and parameters
    cv = GridSearchCV(pipe, param_grid=parameters, cv=2, verbose=10)
    
    return cv


def evaluate_model(cv, X_test, y_test, category_names):
    """
    Function to evaluate model
    """
    
    # Predict results of X_test
    y_pred = cv.predict(X_test)
    
    
    
    # Converting both y_pred and Y_test into DataFrames
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    y_test = pd.DataFrame(y_test, columns=category_names)
    
    
    # Print classification report and accuracy with respect to each column
    for c in category_names:
        print(c, classification_report(y_test[c].values, y_pred[c].values))
        print("Accuracy of "+str(c)+": "+str(accuracy_score(y_test[c].values, y_pred[c].values)))
    


def save_model(model, model_filepath):
    """
    This function for saving the model
    """
    
    # open the file
    pickle_out = open(model_filepath, "wb")
    
    # write model to it
    pickle.dump(model, pickle_out)
    
    # close pickle file
    pickle_out.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
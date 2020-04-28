import json
import plotly
import pandas as pd

import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    
    tokens = word_tokenize(text)
    
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extracting data needed for visuals
    
    # preparation of first visual: Distribution of message Genres
 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # preparation of second visual: Top 5 words (normalized, tokenized & lemmatized)
    
    messages = list(df['message'])
    word_list = []
    for i, message in enumerate(messages):
        msg = tokenize(message)
        for j, wd in enumerate(msg): 
            word_list.append(wd)
    
    word_counts = pd.Series(word_list).value_counts()[:5]
    word_names = list(word_counts.index)
    
    # preparation of third visual: Top 10 message Labels
    
    label_counts = df.iloc[:,4:].sum().sort_values(ascending=False)[:10]
    label_names = list(label_counts.index)
    
    # visuals
    
    graphs = [
        # First visual: Distribution of message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                
            }
        },
        #Second visual: Top 5 words (normalized, tokenized & lemmatized)
        {
            'data': [
                Bar(
                    x=word_names,
                    y=word_counts
                )
            ],

            'layout': {
                'title': 'Top 5 Words (normalized, tokenized & lemmatized)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                },
                
            }
        },
        # Third visual: Top 10 message Labels
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                },
                
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
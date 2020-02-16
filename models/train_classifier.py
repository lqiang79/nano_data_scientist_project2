import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import re
import pickle

def load_data(database_filepath):
    """ load data from give file path

    Args:
        database_filepath {string} -- file path of database
    
    Returns:
        X: {Dataframe} -- dataframe of message and genre
        Y: features matrix of request, offer etc.
        category_names: category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("messages_categories",engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]

    return X, Y, category_names


def tokenize(text):
    """ tokenize word in text

    Args:
        text {string} -- text 

    Returns:
        tokenized words list
    """
    text_regex = '[^0-9a-zA-Z]+'
    find_regex = re.findall(text_regex, text)
    # also remove all empty items in the list.
    final_regex = [x for x in find_regex if x != ' ']
    for rgx in find_regex:
        text = text.replace(rgx, ' ')
    # tokenize and lemmatize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # final output after lowercasing/stripping
    final_token = []
    for t in tokens:
        clean = lemmatizer.lemmatize(t).lower().strip()
        final_token.append(clean)

    return final_token


def build_model():
    """ build model method

    """
    #building a pipeline with 3 steps: vectorize, transform, classify
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators': [30,50,100],
        'clf__estimator__max_depth': [3, 5],
        'clf__estimator__min_samples_split': [3,5],
        'clf__estimator__criterion' : ['gini', 'entropy']
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.fit(X_test)
    for category in range(len(category_names)):
        print('category: {}'.format(category_names[category]))
        print(classification_report(Y_test[:, category], y_pred[:, category]))


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
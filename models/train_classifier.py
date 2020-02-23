import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

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
    df = pd.read_sql_table("messages_categories", engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = df.columns[4:]

    return X, y, category_names


def tokenize(text):
    """ tokenize word in text

    Args:
        text {string} -- text 

    Returns:
        tokenized words list
    """
    # tokenize and lemmatize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # final output after lowercasing/stripping
    final_token = []
    for t in tokens:
        tok = lemmatizer.lemmatize(t).lower().strip()
        final_token.append(tok)
    return final_token


def build_model():
    """ build model method

    Returns: 
        sklearn pipeline of RandomForestClassifier
    """
    # building a pipeline with 3 steps: vectorize, transform, classify
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters)
    return model
    
"""     parameters = {
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators': [30,50,100],
        'clf__estimator__max_depth': [3,5],
        'clf__estimator__min_samples_split': [3,6],
        'clf__estimator__criterion' : ['gini', 'entropy']
    }
    gscvModel = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10)
    return gscvModel """


def evaluate_model(model, X_test, Y_test, category_names):
    """ print out classification_report of model evaluataion

    Args: 
        model: a sklearn model
        X_test: test set of X value
        Y_test: test set of y value
        catagory_names: string, the name of category
    """
    Y_pred = model.predict(X_test)

    # print classification & accuracy score
    print(classification_report(np.hstack(Y_test.values), np.hstack(Y_pred), target_names=category_names))
    print('Accuracy: {}'.format(np.mean(Y_test.values == Y_pred)))

    for idx in range(Y_test.columns.shape[0]):
        test = Y_test[Y_test.columns[idx]].values
        pred = Y_pred[:, idx]

        # calculate different scores
        accu = accuracy_score(test, pred)
        prec = precision_score(test, pred)
        reca = recall_score(test, pred)
        f1_s = f1_score(test, pred)
        # print results
        print(
            f"{Y_test.columns[idx]:22s} | accuracy: {accu:.2f} | precision: {prec:.2f} | recall: {reca:.2f} | f1 score: {f1_s:.2f}")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

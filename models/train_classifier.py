import pickle
import sys
import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')


def load_data(database_filepath):
    """ load data from give file path

    Args:
        database_filepath {string} -- file path of database

    Returns:
        X: {DataFrame} -- dataframe of message and genre
        y: {DataFrame} -- features matrix of request, offer etc.
        category_names: {DataFrame} -- category names
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
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # Reduce words to their root form
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return words


def build_model():
    """ build model method

    Returns: 
        sklearn pipeline of RandomForestClassifier
    """
    # building a pipeline with 3 steps: vectorize, transform, classify
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [30, 40]
    }
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters, cv=5, n_jobs=-1)
    return model


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
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        reports = classification_report(Y_test.iloc[i], Y_pred[i])
        print(f'category: {category}\n {reports}')


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

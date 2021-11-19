#Import required packages
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
import pickle
from sklearn import metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix

from sklearn.tree import DecisionTreeClassifier

multilabel_binarizer = MultiLabelBinarizer()

def verify_null_values(movies_df:DataFrame):
    #Verify if there are any null/NAN values
    print(movies_df.isnull().sum()>0)

def drop_duplicates(movies_df:DataFrame):
    movies_df.drop_duplicates(['synopsis'], inplace=True)

def clean_synopsis(movies_df:DataFrame):
    # Synopsis to be cleaned
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:x.lower(),)
    # Alphabets and may be numbers makes sense for the synopsis for now so clean other characters
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:re.sub("[^a-zA-Z0-9]"," ",x))
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:" ".join(x.split()))

def convert_genres(movies_df:DataFrame):
    # Convert genres to list of genres
    movies_df['genres']=movies_df.genres.apply(lambda x: [genre for genre in x.split()])


def exclude_stopwords_and_lemmatize(movies_df:DataFrame):
    lem = WordNetLemmatizer()
    stemmer=SnowballStemmer(language='english')
    stop_words = set(stopwords.words('english'))
    movies_df['synopsis'] = movies_df['synopsis'].apply(lambda x:' '.join([lem.lemmatize(word)for word in x.split()
                                                                           if not word in stop_words]))


def train_model(file_name):
    train_df = pd.read_csv('./static/datasets/'+file_name, sep=",")
    verify_null_values(train_df)
    drop_duplicates(train_df)
    clean_synopsis(train_df)
    convert_genres(train_df)
    exclude_stopwords_and_lemmatize(train_df)
    # transform target variable
    X = train_df['synopsis']
    y = multilabel_binarizer.fit_transform(train_df['genres'])
    pickle.dump(multilabel_binarizer, open('binarizer.pkl', 'wb'))
    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0, max_features=10000, stop_words="english",
                                       ngram_range=(1, 3))
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))
    # Start training with different classifiers
    train_with_classifiers(x_train_tfidf, y_train, x_test_tfidf, y_test,train_df['genres'])


def train_with_classifiers(x_train,y_train,x_test,y_test,genres):
    lr = LogisticRegression(max_iter=1000)
    nb = MultinomialNB(alpha=0.1)
    rfc=RandomForestClassifier(random_state=5)
    dtc=DecisionTreeClassifier()
    knn=KNeighborsClassifier()

    # clf = OneVsRestClassifier(lr)
    #
    # # fit model on train data
    # clf.fit(x_train, y_train)
    #
    # #Save the Model to pickle file
    # pickle.dump(clf, open('trained_model_onevsrest.pkl', 'wb'))

   # evaluate_model(x_train,y_train,x_test,y_test,clf)
    # grid_search_evaluation(clf,x_train,y_train)
    #binary_relevance(x_train, y_train,x_test,y_test)
    classifier_chains(x_train, y_train, x_test, y_test)
    #label_powerset(x_train, y_train, x_test, y_test)
    #adaped_algorithm(x_train, y_train, x_test, y_test)


def binary_relevance(x_train, y_train,x_test,y_test):
    classifier = BinaryRelevance(GaussianNB())
    # train
    classifier.fit(x_train, y_train)
    pickle.dump(classifier, open('trained_model_binary.pkl', 'wb'))
    # predict
    predictions = classifier.predict(x_test)
    # accuracy
    print(" Binary Relevance Accuracy = ",accuracy_score(y_test,predictions))
    evaluate_model(x_train, y_train, x_test, y_test, classifier)
    print("\n")

def classifier_chains(x_train, y_train,x_test,y_test):
     # using classifier chains
     # initialize classifier chains multi-label classifier
     classifier = ClassifierChain(LogisticRegression(max_iter=1000))
     # Training logistic regression model on train data
     classifier.fit(x_train, y_train)
     pickle.dump(classifier, open('trained_model_classifierchain.pkl', 'wb'))
     # predict
     predictions = classifier.predict(x_test)
     # accuracy
     print(" Classifier chain Accuracy = ",accuracy_score(y_test,predictions))
     evaluate_model(x_train, y_train, x_test, y_test, classifier)
     print("\n")

def label_powerset(x_train, y_train,x_test,y_test):
    classifier = LabelPowerset(LogisticRegression())
    # train
    classifier.fit(x_train, y_train)
    pickle.dump(classifier, open('trained_model_labelpowerset.pkl', 'wb'))
    # predict
    predictions = classifier.predict(x_test)
    # accuracy
    print(" Label Power Set Accuracy = ",accuracy_score(y_test,predictions))
    evaluate_model(x_train, y_train, x_test, y_test, classifier)
    print("\n")

def adaped_algorithm(x_train, y_train,x_test,y_test):
    classifier = MLkNN(k=10)
    # Note that this classifier can throw up errors when handling sparse matrices.
    x_train = lil_matrix(x_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test).toarray()
    # train
    classifier.fit(x_train, y_train)
    pickle.dump(classifier, open('trained_model_adaptedalg.pkl', 'wb'))
    # predict
    predictions= classifier.predict(x_test)
    # accuracy
    print("Adapted Algorithm Accuracy = ",accuracy_score(y_test,predictions))
    evaluate_model(x_train, y_train, x_test, y_test, classifier)
    print("\n")

def predict(file_name):
    predict_df = pd.read_csv('./static/datasets/'+file_name, sep=",")
    predict_df['synopsis_origin']=predict_df['synopsis']
    verify_null_values(predict_df)
    drop_duplicates(predict_df)
    clean_synopsis(predict_df)
    exclude_stopwords_and_lemmatize(predict_df)
    # make predictions for test
    clf = pickle.load(open('trained_model.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    binarizer = pickle.load(open('binarizer.pkl', 'rb'))
    x_test_tfidf= tfidf_vectorizer.transform(predict_df['synopsis'])
    y_pred_test = clf.predict(x_test_tfidf)
    predict_df['predicted_genres'] = binarizer.inverse_transform(y_pred_test)
    predict_df['predicted_genres'] = predict_df['predicted_genres'].apply(lambda x:[elem for elem in x])
    predict_df['predicted_genres'] = predict_df['predicted_genres'].apply(lambda x: " ".join(elem for elem in x))
    predict_df['synopsis'] = predict_df['synopsis_origin']
    predicted_results_df =predict_df[['movie_id','synopsis','predicted_genres']]
    predicted_results_df.to_csv(path_or_buf='./static/datasets/predicted.csv')


def evaluate_model(x_train,y_train,x_test,y_test ,clf):
    # make predictions for test
    #clf = pickle.load(open('trained_model.pkl', 'rb'))
    y_pred = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    print(f1_score(y_test, y_pred_test, average="micro"))
#   print(classification_report(y_test.flatten(), y_pred_test.flatten()))
    print(accuracy_score(y_train, y_pred))
    print(accuracy_score(y_test, y_pred_test))

def grid_search_evaluation(estimator,x_train,y_train):
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    params = {
        'max_features': np.arange(0.1, 1, 0.1).tolist(),  # Number of features to consider as a fraction of all features
        'max_depth': [1, 2, 4, 8, None]  # Depth of the tree
    }
    clf = GridSearchCV(estimator=dtc,
                       param_grid=params,
                       scoring='accuracy',
                       cv=5,
                       verbose=1,
                       n_jobs=-1
                       )
    result = clf.fit(x_train, y_train)
    print(result.best_params_)
    print("The best parameters are :", result.best_params_)
    print("The best accuracy is {:.2f}%:".format(clf.best_score_ * 100))
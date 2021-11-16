#Import required packages
import re
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
import pickle

from sklearn.tree import DecisionTreeClassifier

multilabel_binarizer = MultiLabelBinarizer()

def verify_null_values(movies_df:DataFrame):
    #Verify if there are any null/NAN values
    print(movies_df.isnull().sum())

def drop_duplicates(movies_df:DataFrame):
    movies_df.drop_duplicates(['synopsis'], inplace=True)

def clean_synopsis(movies_df:DataFrame):
    # Synopsis to be cleaned
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:x.lower(),)
    # Alphabets makes sense for the synopsis for now so clean other characters
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:re.sub("[^a-zA-Z0-9]"," ",x))
    movies_df['synopsis']=movies_df.synopsis.apply(lambda x:" ".join(x.split()))

def clean_genres(movies_df:DataFrame):
    # genres to be cleaned
    #movies_df['genres']=movies_df.genres.apply(lambda x:x.lower())
    # Alphabets makes sense for the genres for now so clean other characters
  #  movies_df['genres']=movies_df.genres.apply(lambda x:re.sub("[^a-zA-Z]"," ",x))
    # Convert genres to list of genres
    movies_df['genres']=movies_df.genres.apply(lambda x: [genre for genre in x.split()])


def exclude_stopwords_and_lemmatize(movies_df:DataFrame):
    lem = WordNetLemmatizer()
    stemmer=SnowballStemmer(language='english')
    stop_words = set(stopwords.words('english'))
    movies_df['synopsis'] = movies_df['synopsis'].apply(lambda x: ' '.join([word for word in x.split()]))


def train_model(file_name):
    train_df = pd.read_csv('./static/datasets/'+file_name, sep=",")
    verify_null_values(train_df)
    drop_duplicates(train_df)
    clean_synopsis(train_df)
    clean_genres(train_df)
    exclude_stopwords_and_lemmatize(train_df)
    # transform target variable
    X = train_df['synopsis']
    y = multilabel_binarizer.fit_transform(train_df['genres'])
    pickle.dump(multilabel_binarizer, open('binarizer.pkl', 'wb'))
    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0, max_features=10000, stop_words="english",ngram_range=(1, 3))
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))
    # Start training with different classifiers
    train_with_classifiers(x_train_tfidf, y_train, x_test_tfidf, y_test)


def train_with_classifiers(x_train,y_train,x_test,y_test):
    lr = LogisticRegression()
    nb = MultinomialNB(alpha=0.1)
    rfc=RandomForestClassifier(random_state=5)
    decisionTree=DecisionTreeClassifier(random_state=5)

    clf = OneVsRestClassifier(lr)

    # fit model on train data
    clf.fit(x_train, y_train)

    #Save the Model to pickle file
    pickle.dump(clf, open('trained_model.pkl', 'wb'))
    y_pred_test = clf.predict(x_test)
    print(multilabel_binarizer.inverse_transform(y_pred_test)[0],multilabel_binarizer.inverse_transform(y_test)[0])

def predict(file_name):
    predict_df = pd.read_csv('./static/datasets/'+file_name, sep=",")
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
    predict_df['predicted_genres'] = predict_df['predicted_genres'].apply(lambda x:"".join(el for elem in x for el in elem ))
    predict_df.to_csv(path_or_buf='./static/datasets/predicted.csv')


def evaluate_model(x_test,y_test):
    # make predictions for test
    clf = pickle.load(open('trained_model.pkl', 'rb'))
    y_pred_test = clf.predict(x_test)
    print(multilabel_binarizer.inverse_transform(y_pred_test)[1], multilabel_binarizer.inverse_transform(y_test))
    print(f1_score(y_test, y_pred_test, average="micro"))
    print(classification_report(y_test.flatten(), y_pred_test.flatten()))

def grid_search_evaluation(x_train_tfidf,y_train):
    decision_tree = DecisionTreeClassifier(random_state=5)
    params = {
        'max_features': np.arange(0.1, 1, 0.1).tolist(),  # Number of features to consider as a fraction of all features
        'max_depth': [1, 2, 4, 8, None]  # Depth of the tree
    }
    clf = GridSearchCV(estimator=decision_tree,
                       param_grid=params,
                       scoring='accuracy',
                       cv=5,
                       verbose=1,
                       n_jobs=-1
                       )

    # As we are doing cross-validation on the training set, the testing set X_test is untouched
    result = clf.fit(x_train_tfidf, y_train)
    print(result.best_params_)
    print("The best parameters are :", result.best_params_)
    print("The best accuracy is {:.2f}%:".format(clf.best_score_ * 100))
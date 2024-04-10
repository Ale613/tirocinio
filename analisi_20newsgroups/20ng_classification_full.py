import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb
from warnings import simplefilter
from sklearn.preprocessing import LabelEncoder

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def calculateResults(nameClassifier, y_test, y_pred, file):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division = 0)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    file.write('\n\n'+ str(nameClassifier)+ ' Results:\n')
    file.write("Accuracy: " + str(accuracy) + "\nPrecision: " + str(precision) + "\nRecall: " + str(recall) + "\nF1: " + str(f1))

def classificazione(X_train, X_test, y_train, y_test, file):

    # training KNN
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    calculateResults(nameClassifier='KNN', y_test=y_test, y_pred=y_pred_knn, file=file)

    # training SVM
    sv_classifier = svm.SVC(random_state=42)
    sv_classifier.fit(X_train, y_train)
    y_pred_svm = sv_classifier.predict(X_test)
    calculateResults(nameClassifier='SVM', y_test=y_test, y_pred=y_pred_svm, file=file)

    # training Tree
    tree_classifier = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_classifier.fit(X_train, y_train)
    y_pred_tree = tree_classifier.predict(X_test)
    calculateResults(nameClassifier='DecisionTree', y_test=y_test, y_pred=y_pred_tree, file=file)

    # training MLP
    mlp_classifier = neural_network.MLPClassifier(alpha=1, max_iter=300, random_state=42)
    mlp_classifier.fit(X_train, y_train)
    y_pred_mlp = mlp_classifier.predict(X_test)
    calculateResults(nameClassifier='MLP', y_test=y_test, y_pred=y_pred_mlp, file=file)

    # training Naive
    if (not(isinstance(X_test, np.ndarray))):
        X_test_nb = X_test.toarray()
        X_train_nb = X_train.toarray()
    else:
        X_test_nb = X_test
        X_train_nb = X_train
    nb_classifier = naive_bayes.GaussianNB()
    nb_classifier.fit(X_train_nb, y_train)
    y_pred_nb = nb_classifier.predict(X_test_nb)
    calculateResults(nameClassifier='Naive', y_test=y_test, y_pred=y_pred_nb, file=file)

    # training AdaBoost
    ada_classifier = AdaBoostClassifier(random_state=42)
    ada_classifier.fit(X_train, y_train)
    y_pred_ada = ada_classifier.predict(X_test)
    calculateResults(nameClassifier='AdaBoost', y_test=y_test, y_pred=y_pred_ada, file=file)

    # training RandomForest
    rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    calculateResults(nameClassifier='RandomForest', y_test=y_test, y_pred=y_pred_rf, file=file)

    # training xgboost
    le = LabelEncoder()
    y_train_xb = le.fit_transform(y_train)
    y_test_xb = le.fit_transform(y_test)
    xb = xgb.XGBClassifier()
    xb.fit(X_train, y_train_xb)
    y_pred_xb = xb.predict(X_test)
    calculateResults(nameClassifier='Xgboost', y_test=y_test, y_pred=y_pred_xb, file=file)

# lettura train.csv
df_20ngTrain = pd.read_csv('analisi_20newsgroups/newsgroups_train.csv', names = ['data', 'target'], sep=',')

# lettura test.csv (fare test completo)
df_20ngTest = pd.read_csv('analisi_20newsgroups/newsgroups_test.csv', names = ['data', 'target'], sep=',')

# train
X_train = df_20ngTrain['data']
y_train = df_20ngTrain['target']
# test
X_test = df_20ngTest['data']
y_test = df_20ngTest['target']

#TD-IDF Full Test
f = open("results_TFIDF_20newsgroups_full.txt", "w")
vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
X_train_tfidf = vectorizer.fit_transform(X_train.values.astype('U'))
X_test_tfidf = vectorizer.transform(X_test.values.astype('U'))
classificazione(X_train=X_train_tfidf, X_test=X_test_tfidf, y_train=y_train, y_test=y_test, file=f)
f.close()

# RoBERTa Full Test
f = open("results_RoBERTa_20newsgroups_full.txt", "w")
model = SentenceTransformer("all-distilroberta-v1")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, X_test=embeddingsTest, y_train=y_train, y_test=y_test, file=f)
f.close()

# mpnet-base Full Test
f = open("results_mpnet_20newsgroups_full.txt", "w")
model = SentenceTransformer("all-mpnet-base-v2")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, X_test=embeddingsTest, y_train=y_train, y_test=y_test, file=f)
f.close()

# ALBERT Full Test
f = open("results_ALBERT_20newsgroups_full.txt", "w")
model = SentenceTransformer("paraphrase-albert-small-v2")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, X_test=embeddingsTest, y_train=y_train, y_test=y_test, file=f)
f.close()
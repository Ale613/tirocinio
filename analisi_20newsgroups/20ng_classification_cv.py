import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.model_selection._validation import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
from sklearn.metrics import make_scorer
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

def calculateResults(nameClassifier, risultati, file):
    accuracy_mean = np.mean(list(risultati['test_accuracy']))
    precision_mean = np.mean(list(risultati['test_precision']))
    recall_mean = np.mean(list(risultati['test_recall']))
    f1_mean = np.mean(list(risultati['test_f1_score']))
    file.write('\n\n'+ str(nameClassifier)+ ' Results:\n')
    file.write("Accuracy: " + str(accuracy_mean) + "\nPrecision: " + str(precision_mean) + "\nRecall: " + str(recall_mean) + "\nF1: " + str(f1_mean))

def classificazione(X_train, y_train, scoring, file):

    # training KNN
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=10)
    risultati_knn = cross_validate(knn_classifier, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='KNN', risultati=risultati_knn, file=file)

    # training SVM (Support vector machine)
    sv_classifier = svm.SVC(random_state=42)
    risultati_svm = cross_validate(sv_classifier, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='SVM', risultati=risultati_svm, file=file)

    # training Tree
    tree_classifier = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
    risultati_tree = cross_validate(tree_classifier, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='DecisionTree', risultati=risultati_tree, file=file)

    # training MLP (Multi-layer Perceptron)
    mlp_classifier = neural_network.MLPClassifier(alpha=1, max_iter=300, random_state=42)
    risultati_mlp = cross_validate(mlp_classifier, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='MLP', risultati=risultati_mlp, file=file)

    # training Naive
    if (not(isinstance(X_train, np.ndarray))):
        X_train_nb = X_train.toarray()
    else:
        X_train_nb = X_train
    nb_classifier = naive_bayes.GaussianNB()
    risultati_nb = cross_validate(nb_classifier, X_train_nb, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='Naive', risultati=risultati_nb, file=file)

    # training AdaBoost
    ada_classifier = AdaBoostClassifier(random_state=42)
    risultati_ada = cross_validate(ada_classifier, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='AdaBoost', risultati=risultati_ada, file=file)

    # training RandomForest
    rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
    risultati_rf = cross_validate(rf, X_train, y_train, scoring=scoring, cv=10)
    calculateResults(nameClassifier='RandomForest', risultati=risultati_rf, file=file)

    # training xgboost
    le = LabelEncoder()
    y_train_xb = le.fit_transform(y_train)
    xb = xgb.XGBClassifier()
    risultati_xb = cross_validate(xb, X_train, y_train_xb, scoring=scoring, cv=10)
    calculateResults(nameClassifier='Xgboost', risultati=risultati_xb, file=file)

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

scoring = {'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average = 'weighted', zero_division = 0),
        'recall' : make_scorer(recall_score, average = 'weighted'),
        'f1_score' : make_scorer(f1_score, average = 'weighted')}

#TD-IDF Cross Validation
f = open("results_TFIDF_20newsgroups_cv.txt", "w")
vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
X_train_tfidf = vectorizer.fit_transform(X_train.values.astype('U'))
X_test_tfidf = vectorizer.transform(X_test.values.astype('U'))
classificazione(X_train=X_train_tfidf, y_train=y_train, scoring=scoring, file=f)
f.close()

# RoBERTa Cross Validation
f = open("results_RoBERTa_20newsgroups_cv.txt", "w")
model = SentenceTransformer("all-distilroberta-v1")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, y_train=y_train, scoring=scoring, file=f)
f.close()

# mpnet-base Cross Validation
f = open("results_mpnet_20newsgroups_cv.txt", "w")
model = SentenceTransformer("all-mpnet-base-v2")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, y_train=y_train, scoring=scoring, file=f)
f.close()

# ALBERT Cross Validation
f = open("results_ALBERT_20newsgroups_cv.txt", "w")
model = SentenceTransformer("paraphrase-albert-small-v2")
embeddingsTrain = model.encode(X_train.values.astype('U'))
embeddingsTest = model.encode(X_test.values.astype('U'))
classificazione(X_train=embeddingsTrain, y_train=y_train, scoring=scoring, file=f)
f.close()
import csv
import numpy as np
import math
from keras.preprocessing.text import Tokenizer
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from threading import Thread
import matplotlib.pyplot as plt
pathname = 'dataset1_2.csv'
pathname1 = 'dataset2_2.csv'
pathname2 = 'dataset3_2.csv'


def csvdata():
    try:
        with open(pathname, "r") as inp:
            words_dict = []
            dict_list = []
            length = []
            target = []
            inp.seek(0)
            lastrow = None
            n = 0
            for lastrow in csv.reader(inp):
                if(n==0):
                    firstrow = lastrow
                n = n + 1
            #print(firstrow)
            inp.seek(0)
            for row in csv.reader(inp):
                words = []
                if(int(row[1]) == -1):
                    target.append(0)
                else:
                    target.append(int(row[1]))
                length.append(row[2])
                for i in range(3,len(row)):
                    dict_list.append(row[i])
                    words.append(row[i])
                words_dict.append(words)
            return dict_list,words_dict,target,length

    except Exception as error:
        print(error)
        return None

def csvdata1():
    try:
        with open(pathname1, "r") as inp:
            trend = []
            inp.seek(0)
            lastrow = None
            n = 0
            for lastrow in csv.reader(inp):
                if (n == 0):
                    firstrow = lastrow
                n = n + 1
            # print(firstrow)
            inp.seek(0)
            for row in csv.reader(inp):

                trend_list = []
                for i in range(5, len(row)):
                    trend_list.append(int(row[i]))
                trend.append(trend_list)
            return trend

    except Exception as error:
        print(error)
        return None

def csvdata2():
    ranking_list = []
    avg = []
    try:
        with open(pathname2, "r") as inp:
            inp.seek(0)
            for row in csv.reader(inp):
                num = (len(row)-3)
                rank = []
                for i in range (1,int(num)+1):
                    val = row[-i]
                    rank.append(int(val))
                ranking_list.append(rank)

            for ele in ranking_list:
                if(-1 in ele):
                    avg.append(-1000.0)
                else:
                    avg.append(sum(ele)/len(ele))
            return avg

    except Exception as error:
        print(error)
        return None


def tfidf(input_):
    count_vectorizer = TfidfVectorizer(stop_words='english',analyzer='word')
    count_vectorizer.fit(input_)
    return count_vectorizer.vocabulary_

def toOnehotkey(input_):
    keydict = {}
    tokenizer = Tokenizer(num_words=len(input_))
    tokenizer.fit_on_texts(input_)
    one_hot_results = tokenizer.texts_to_matrix(input_, mode='binary')
    for key,val in zip(input_,one_hot_results):
        keydict[key] = val
    return  keydict

def calcTrends(trends):
    trends_ratio_15 = []
    trends_ratio_30 = []
    trends_ratio_60 = []
    coef = []
    for ele in trends:
        trends_ratio_15.append(trend_ratio_n(ele, 15))
        trends_ratio_30.append(trend_ratio_n(ele, 30))
        trends_ratio_60.append(trend_ratio_n(ele, 60))
        x = np.array([i for i in range(len(ele))]).reshape(-1, 1)
        y = np.array(ele).reshape(-1, 1)
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(x, y)
        coef.append((reg.coef_[0][0]))

    return trends_ratio_15,trends_ratio_30,trends_ratio_60,coef

def trend_ratio_n(trend, n):
    trend_ratio = []
    avg_0 = sum(trend[:n]) / len(trend[:n])
    if(avg_0 == 0):
        avg_0 = 0.1
    for i in range(1,int((len(trend)-1)/n)):
        avg_1 = sum(trend[(i*n):(i*n+n)]) / len(trend[(i*n):(i*n+n)])
        trend_ratio.append(avg_1/avg_0)
    trend_ratio.append(trend[-1]/avg_0)
    return trend_ratio

def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return h

def inputMatrix(dict_list, words_dict, trends_ratio_15,trends_ratio_30,trends_ratio_60,coef,length_domains):
    maxval = 0
    X_ = []
    hotkey_dict = toOnehotkey(dict_list)
    tfidf_ = tfidf(dict_list)
    """
    for ele in words_dict:
        if maxval < len(ele):
            maxval = len(ele)
    """
    maxval = 6
    for ele1 in words_dict:
        length = len(ele1)
        arr = hotkey_dict[ele1[0]]
        if (ele1[0] in tfidf_.keys()):
            arr1 = [tfidf_[ele1[0]]]
        else:
            arr1 = [-1000]
        for i in range(1, maxval):
            # print(i%length)
            arr = np.concatenate([arr, hotkey_dict[ele1[i % length]]])
            if (ele1[i % length] in tfidf_.keys()):
                arr1 = np.concatenate([arr1, [tfidf_[ele1[i % length]]]])
            else:
                arr1 = np.concatenate([arr1, [-1000]])
        X_.append(np.concatenate([arr, arr1]))
    X_ = np.concatenate((X_,trends_ratio_15,trends_ratio_30,trends_ratio_60,np.array(coef).reshape(-1, 1),np.array(length_domains).reshape(-1, 1)), axis=1)
    return X_.astype(float)

def train_predict(X_train, X_test, y_train, y_test,tag):
    model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest','1-layer Neural Network']
    model = LogisticRegression(C=1e5, random_state=0, solver='liblinear', multi_class='auto', max_iter=5000)
    model1 = DecisionTreeClassifier(criterion='entropy', max_depth=21, random_state=0)
    model2 = SVC(kernel='poly', tol=1e-3, random_state=0, gamma="scale", C=1, verbose=True)
    # model2 = SVC(kernel='linear')
    model3 = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
    model4 = RandomForestClassifier(criterion='entropy', n_estimators=21, random_state=1, n_jobs=2)
    model5 = nn_model(np.shape(X_train)[1])
    model_list = [model,model1,model2,model3,model4,model5]
    re_model_list = []
    train_acc_list = []
    test_acc_list = []
    test_neg_acc_list = []

    print("\n\n\n\n--------------Train "+tag+"-----------------")
    for mod,name in zip(model_list,model_name_list):
        if(name == '1-layer Neural Network'):
            features = X_train
            targets = np.array(keras.utils.to_categorical(y_train))
            features_test = X_test
            targets_test = np.array(keras.utils.to_categorical(y_test))
            mod.fit(features, targets, epochs=10, batch_size=32, validation_data=(features_test, targets_test),
                    verbose=0)
            # Evaluating the model on the training and testing set
            score = mod.evaluate(features, targets)
            print("\n Training Accuracy:", score[1])
            score = mod.evaluate(features_test, targets_test)
            print("\n Testing Accuracy:", score[1])
            prediction_y = np.argmax(mod.predict(features_test), axis=-1)
            train_prediction_y = np.argmax(mod.predict(features), axis=-1)
        else:
            mod.fit(X_train, y_train)
            prediction_y = mod.predict(X_test)
            train_prediction_y = mod.predict(X_train)
        test = []
        predict = []
        j = 0
        for ele in y_test:
            if(ele == 0):
                test.append(0)
                predict.append(prediction_y[j])
            j = j+1
        re_model_list.append(mod)
        train_accuracy = accuracy_score(y_train, train_prediction_y)
        test_accuracy = accuracy_score(y_test, prediction_y)
        neg_acc = accuracy_score(test, predict)
        train_acc_list.append(train_accuracy)
        test_neg_acc_list.append(neg_acc)
        test_acc_list.append(test_accuracy)
        print(prediction_y)
        print(y_test)
        print("The accuracy of "+name+" model is: ",test_accuracy)
        print("The negative accuracy of "+name+" model is: ",neg_acc)

    return re_model_list, train_acc_list, test_acc_list, test_neg_acc_list

def predict_exist(model_list, X_test, y_test):
    model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest','1-layer Neural Network']
    print("\n\n\n\n--------------k-fold model Predict-----------------")
    for mod, name in zip(model_list, model_name_list):
        if (name == '1-layer Neural Network'):
            features_test = X_test
            prediction_y = np.argmax(mod.predict(features_test), axis=-1)
        else:
            prediction_y = mod.predict(X_test)
        test = []
        predict = []
        j = 0
        for ele in y_test:
            if (ele == 0):
                test.append(0)
                predict.append(prediction_y[j])
            j = j + 1
        accuracy = accuracy_score(y_test, prediction_y)
        neg_acc = accuracy_score(test, predict)
        print(prediction_y)
        print(y_test)
        print("The accuracy of " + name + " model is: ", accuracy)
        print("The negative accuracy of " + name + " model is: ", neg_acc)

def train_predict_exist(model_list, X_train, X_test, y_train, y_test):
    model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest','1-layer Neural Network']
    print("\n\n\n\n--------------k-fold model Train and Predict-----------------")
    for mod,name in zip(model_list,model_name_list):
        if(name == '1-layer Neural Network'):
            features = X_train
            targets = np.array(keras.utils.to_categorical(y_train))
            features_test = X_test
            targets_test = np.array(keras.utils.to_categorical(y_test))
            mod.fit(features, targets, epochs=10, batch_size=32, validation_data=(features_test, targets_test),
                    verbose=0)
            # Evaluating the model on the training and testing set
            score = mod.evaluate(features, targets)
            print("\n Training Accuracy:", score[1])
            score = mod.evaluate(features_test, targets_test)
            print("\n Testing Accuracy:", score[1])
            prediction_y = np.argmax(mod.predict(features_test), axis=-1)

        else:
            mod.fit(X_train, y_train)
            prediction_y = mod.predict(X_test)
        test = []
        predict = []
        j = 0
        for ele in y_test:
            if(ele == 0):
                test.append(0)
                predict.append(prediction_y[j])
            j = j+1
        accuracy = accuracy_score(y_test, prediction_y)
        neg_acc = accuracy_score(test, predict)
        print(prediction_y)
        print(y_test)
        print("The accuracy of "+name+" model is: ",accuracy)
        print("The negative accuracy of "+name+" model is: ",neg_acc)
def nn_model(input_dim ,dropout_rate=0.2,neurons=128):
    print("input size:", input_dim)
    # Building the model
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def learning_rate(X, y, n_samples=20, cv = 5):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RepeatedKFold
    #skf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    random_state = 12883823
    rkf = RepeatedKFold(n_splits=cv, n_repeats=10, random_state=random_state)
    X_plot = []

    pos_train_mean_list = []
    pos_test_mean_list = []
    neg_train_mean_list = []
    neg_test_mean_list = []
    pos_train_std_list = []
    pos_test_std_list = []
    neg_train_std_list = []
    neg_test_std_list = []
    pos_train_confidence_list = []
    pos_test_confidence_list = []
    neg_train_confidence_list = []
    neg_test_confidence_list = []


    for pos in (np.linspace(0.02, 0.10, num=n_samples)):
        n_sample = int(len(y)*pos)
        print("\n\n\n\n-------------- %d samples-----------------" % n_sample)
        X_plot.append(n_sample)
        X_sample = X[:n_sample]
        y_sample = y[:n_sample]

        pos_train_score = {}
        pos_test_score = {}
        neg_test_score = {}
        neg_train_score = {}
        isExist = 0
        model5 = nn_model(np.shape(X_sample)[1])

        model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest','1-layer Neural Network']
        for train_index, val_index in rkf.split(X_sample, y_sample):

            # print("Train:", train_index, "Validation:", val_index)
            X_train, X_test = X[train_index], X[val_index]
            y_train, y_test = y[train_index], y[val_index]

            model = LogisticRegression(C=1e5, random_state=0, solver='liblinear', multi_class='auto', max_iter=5000)
            model1 = DecisionTreeClassifier(criterion='entropy', max_depth=21, random_state=0)
            model2 = SVC(kernel='poly', tol=1e-3, random_state=0, gamma="scale", C=1, verbose=True)
            # model2 = SVC(kernel='linear')
            model3 = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
            model4 = RandomForestClassifier(criterion='entropy', n_estimators=21, random_state=1, n_jobs=2)
            model_list = [model, model1, model2, model3, model4, model5]

            pos_train_accuracy = []
            pos_test_accuracy = []
            neg_test_acc = []
            neg_train_acc = []


            print('Train: %s | test: %s' % (train_index, val_index))
            for mod, name in zip(model_list, model_name_list):
                if(isExist < len(model_name_list)):
                    neg_test_score[name] = []
                    neg_train_score[name] = []
                    pos_train_score[name] = []
                    pos_test_score[name] = []
                    isExist = isExist + 1

                if(name == '1-layer Neural Network'):
                    features = X_train
                    targets = np.array(keras.utils.to_categorical(y_train))
                    features_test = X_test
                    targets_test = np.array(keras.utils.to_categorical(y_test))
                    mod.fit(features, targets, epochs=10, batch_size=32, validation_data=(features_test, targets_test), verbose=0)
                    # Evaluating the model on the training and testing set
                    score = mod.evaluate(features, targets)
                    print("\n Training Accuracy:", score[1])
                    score = mod.evaluate(features_test, targets_test)
                    print("\n Testing Accuracy:", score[1])

                    test_prediction_y = np.argmax(mod.predict(features_test), axis=-1)
                    train_prediction_y = np.argmax(mod.predict(features), axis=-1)
                else:
                    mod.fit(X_train, y_train)
                    test_prediction_y = mod.predict(X_test)
                    train_prediction_y = mod.predict(X_train)
                neg_test = []
                neg_train = []
                neg_predict_test = []
                neg_predict_train = []
                j = 0
                for ele, ele1 in zip(y_test, y_train):
                    if (ele == 0):
                        neg_test.append(0)
                        neg_predict_test.append(test_prediction_y[j])
                    if (ele1 == 0):
                        neg_train.append(0)
                        neg_predict_train.append(train_prediction_y[j])
                    j = j + 1
                # print(y_train_, y_test, neg_test, neg_predict_test)
                pos_train_accuracy.append(balanced_accuracy_score(y_train, train_prediction_y, adjusted=True))
                pos_test_accuracy.append(balanced_accuracy_score(y_test, test_prediction_y, adjusted=True))
                neg_test_acc.append(accuracy_score(neg_test, neg_predict_test))
                neg_train_acc.append(accuracy_score(neg_train, neg_predict_train))

            for re,name in zip(neg_test_acc,model_name_list):neg_test_score[name].append(re)
            for re,name in zip(neg_train_acc,model_name_list):neg_train_score[name].append(re)
            for re,name in zip(pos_test_accuracy,model_name_list):pos_test_score[name].append(re)
            for re,name in zip(pos_train_accuracy,model_name_list):pos_train_score[name].append(re)

        #print(neg_test_score, neg_train_score, pos_test_score, pos_train_score)
        neg_test_score_mean = [np.nanmean(ele) for ele in neg_test_score.values()]
        neg_train_score_mean = [np.nanmean(ele) for ele in neg_train_score.values()]
        pos_test_score_mean = [np.nanmean(ele) for ele in pos_test_score.values()]
        pos_train_score_mean = [np.nanmean(ele) for ele in pos_train_score.values()]


        neg_test_score_confidence = [mean_confidence_interval(ele) for ele in neg_test_score.values()]
        neg_train_score_confidence = [mean_confidence_interval(ele) for ele in neg_train_score.values()]
        pos_test_score_confidence = [mean_confidence_interval(ele) for ele in pos_test_score.values()]
        pos_train_score_confidence = [mean_confidence_interval(ele) for ele in pos_train_score.values()]


        neg_test_score_std = [np.nanstd(ele) for ele in neg_test_score.values()]
        neg_train_score_std = [np.nanstd(ele) for ele in neg_train_score.values()]
        pos_test_score_std = [np.nanstd(ele) for ele in pos_test_score.values()]
        pos_train_score_std = [np.nanstd(ele) for ele in pos_train_score.values()]

        pos_train_confidence = {}
        pos_test_confidence = {}
        neg_train_confidence = {}
        neg_test_confidence = {}

        pos_train_mean = {}
        pos_test_mean = {}
        neg_train_mean = {}
        neg_test_mean = {}

        pos_train_std = {}
        pos_test_std = {}
        neg_train_std = {}
        neg_test_std = {}

        for name,val1,val2,val3,val4,ele1,ele2,ele3,ele4,value1,value2,value3,value4 in zip(model_name_list,neg_train_score_mean,neg_test_score_mean, pos_train_score_mean, pos_test_score_mean
                , neg_train_score_std,neg_test_score_std, pos_train_score_std, pos_test_score_std, neg_test_score_confidence, neg_train_score_confidence, pos_test_score_confidence, pos_train_score_confidence):
            neg_train_mean[name] = val1
            neg_test_mean[name] = val2
            pos_train_mean[name] = val3
            pos_test_mean[name] = val4

            neg_train_std[name] = ele1
            neg_test_std[name] = ele2
            pos_train_std[name] = ele3
            pos_test_std[name] = ele4

            neg_train_confidence[name] = value2
            neg_test_confidence[name] = value1
            pos_train_confidence[name] = value4
            pos_test_confidence[name] = value3


        pos_train_std_list.append(pos_train_std)
        pos_test_std_list.append(pos_test_std)
        neg_train_std_list.append(neg_train_std)
        neg_test_std_list.append(neg_test_std)
        pos_train_mean_list.append(pos_train_mean)
        pos_test_mean_list.append(pos_test_mean)
        neg_train_mean_list.append(neg_train_mean)
        neg_test_mean_list.append(neg_test_mean)

        pos_train_confidence_list.append(pos_train_confidence)
        pos_test_confidence_list.append(pos_test_confidence)
        neg_train_confidence_list.append(neg_train_confidence)
        neg_test_confidence_list.append(neg_test_confidence)

    plt_dict = {}
    for name in model_name_list:
        y_1_mean = []
        y_2_mean = []
        y_3_mean = []
        y_4_mean = []
        y_1_std = []
        y_2_std = []
        y_3_std = []
        y_4_std = []
        y_1_confidence = []
        y_2_confidence = []
        y_3_confidence = []
        y_4_confidence = []

        for ele,ele1,ele2,ele3,val1,val2,val3,val4,value1,value2,value3,value4 in zip(pos_train_mean_list,pos_test_mean_list,neg_train_mean_list,neg_test_mean_list
                ,pos_train_std_list,pos_test_std_list,neg_train_std_list,neg_test_std_list,pos_train_confidence_list,pos_test_confidence_list,neg_train_confidence_list,neg_test_confidence_list):
            y_1_mean.append(ele[name])
            y_2_mean.append(ele1[name])
            y_3_mean.append(ele2[name])
            y_4_mean.append(ele3[name])
            y_1_std.append(val1[name])
            y_2_std.append(val2[name])
            y_3_std.append(val3[name])
            y_4_std.append(val4[name])

            y_1_confidence.append(value1[name])
            y_2_confidence.append(value2[name])
            y_3_confidence.append(value3[name])
            y_4_confidence.append(value4[name])


        plt_dict[name] = [y_2_mean,y_4_mean,y_2_std,y_4_std,y_2_confidence,y_4_confidence]
    print(plt_dict)
    fig, ax = plt.subplots()
    colors = ['b','g','r','c','m','y']
    for name, color in zip(model_name_list,colors):
        #line1, = ax.plot(X_plot, plt_dict[name], 'c*-', label='balanced train score')
        line2, = ax.plot(X_plot, plt_dict[name][0], 'm.-.', label='balanced cross-validation score:'+name, c = color)
        ax.fill_between(X_plot, np.array(plt_dict[name][0]) - np.array(plt_dict[name][2]),
                         np.array(plt_dict[name][0]) + np.array(plt_dict[name][2]), alpha=0.1,
                         color=color)
        #line3, = ax.plot(X_plot, plt_dict[name], 'b*-', label='negative train score')
        #line4, = ax.plot(X_plot, plt_dict[name][1], 'r.-.', label='negative cross-validation score:'+name)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('cross-validation examples')
    ax.set_ylabel('balanced accuracy')
    ax.set_title('Learning curves (+/- standard variance)')
    ax.legend()
    plt.grid()
    fig.savefig('output/balanced_score_std.png')

    fig, ax = plt.subplots()
    for name, color in zip(model_name_list,colors):
        #line1, = ax.plot(X_plot, plt_dict[name][0], 'c*-', label='balanced train score:'+name)
        #line2, = ax.plot(X_plot, plt_dict[name], 'm.-.', label='balanced cross-validation score')
        #line3, = ax.plot(X_plot, plt_dict[name], 'b*-', label='negative train score')
        line4, = ax.plot(X_plot, plt_dict[name][1], 'r.-.', label='negative cross-validation score:'+name, c = color)
        ax.fill_between(X_plot, np.array(plt_dict[name][1]) - np.array(plt_dict[name][3]),
                        np.array(plt_dict[name][1]) + np.array(plt_dict[name][3]), alpha=0.1,
                        color=color)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('cross-validation examples')
    ax.set_ylabel('negative accuracy')
    ax.set_title('Learning curves (+/- standard variance)')
    ax.legend()
    plt.grid()
    fig.savefig('output/negative_score_std.png')


    fig, ax = plt.subplots()
    for name, color in zip(model_name_list,colors):
        #line1, = ax.plot(X_plot, plt_dict[name], 'c*-', label='balanced train score')
        line2, = ax.plot(X_plot, plt_dict[name][0], 'm.-.', label='balanced cross-validation score:'+name, c = color)
        ax.fill_between(X_plot, np.array(plt_dict[name][0]) - np.array(plt_dict[name][4]),
                         np.array(plt_dict[name][0]) + np.array(plt_dict[name][4]), alpha=0.1,
                         color=color)
        #line3, = ax.plot(X_plot, plt_dict[name], 'b*-', label='negative train score')
        #line4, = ax.plot(X_plot, plt_dict[name][1], 'r.-.', label='negative cross-validation score:'+name)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('cross-validation examples')
    ax.set_ylabel('balanced accuracy')
    ax.set_title('Learning curves (+/- confidence interval)')
    ax.legend()
    plt.grid()
    fig.savefig('output/balanced_score_confidence.png')

    fig, ax = plt.subplots()
    for name, color in zip(model_name_list,colors):
        #line1, = ax.plot(X_plot, plt_dict[name][0], 'c*-', label='balanced train score:'+name)
        #line2, = ax.plot(X_plot, plt_dict[name], 'm.-.', label='balanced cross-validation score')
        #line3, = ax.plot(X_plot, plt_dict[name], 'b*-', label='negative train score')
        line4, = ax.plot(X_plot, plt_dict[name][1], 'r.-.', label='negative cross-validation score:'+name, c = color)
        ax.fill_between(X_plot, np.array(plt_dict[name][1]) - np.array(plt_dict[name][5]),
                        np.array(plt_dict[name][1]) + np.array(plt_dict[name][5]), alpha=0.1,
                        color=color)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('cross-validation examples')
    ax.set_ylabel('negative accuracy')
    ax.set_title('Learning curves (+/- confidence interval)')
    ax.legend()
    plt.grid()
    fig.savefig('output/negative_score_confidence.png')



"""
def learning_rate(X, y):
    from sklearn.model_selection import learning_curve
    model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest']
    model = LogisticRegression(C=1e5, random_state=0, solver='liblinear', multi_class='auto', max_iter=5000)
    model1 = DecisionTreeClassifier(criterion='entropy', max_depth=21, random_state=0)
    model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=1e5, verbose=True)
    # model2 = SVC(kernel='linear')
    model3 = KNeighborsClassifier(n_neighbors=21, p=3, metric='minkowski')
    model4 = RandomForestClassifier(criterion='entropy', n_estimators=21, random_state=1, n_jobs=2)
    model_list = [model, model1, model2, model3, model4]
    plot_re_ = {}
    for name,model_ele in zip(model_name_list,model_list):
        train_sizes, train_scores, valid_scores = learning_curve(model_ele, X, y, train_sizes = [0.05, 0.1, 0.33, 0.55, 0.78, 1. ], cv = 5)
        train_scores = [sum(row)/len(row) for row in train_scores]
        valid_scores = [sum(row)/len(row) for row in valid_scores]
        plot_re_[name] = [train_sizes, train_scores, valid_scores]
    for name in  model_name_list:
        X = plot_re_[name][0]
        y1 = plot_re_[name][1]
        y2 = plot_re_[name][2]


        fig, ax = plt.subplots()
        line1, = ax.plot(X, y1,'c*-', label='train score' )
        line2, = ax.plot( X, y2, 'm.-.', label='cross-validation score')
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel('training examples')
        ax.set_ylabel('accuracy')
        ax.set_title('Learning curves: '+name)
        ax.legend()
        plt.show()
        fig.savefig('output/'+name+'.png')
"""


def grid_search(X_train_, X_test_, y_train_, y_test_):
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from keras.wrappers.scikit_learn import KerasClassifier

    scaler = StandardScaler()
    X_transform = scaler.fit_transform(X_train_)

    #parameters_svm = {'kernel': ('linear',  'poly', 'rbf'), 'C': [1, 10, 100, 1e5]}
    parameters_svm = {'kernel': ['linear'], 'C': [1]}

    #parameters_logistic = {'solver': ('liblinear', 'saga'), 'C': [1, 10, 100, 1e5], 'max_iter': [1000, 2000, 3000]}
    parameters_logistic = {'solver': ['liblinear'], 'C': [1], 'max_iter': [1000]}

    #parameters_decisiontree = {'criterion': ('entropy', 'gini'), 'max_depth': [10, 21, 42]}
    parameters_decisiontree = {'criterion': ['entropy'], 'max_depth': [10]}

    #parameters_kneighbors = {'n_neighbors': (10, 15, 21, 27), 'p': (1, 2, 3)}
    parameters_kneighbors = {'n_neighbors': [10], 'p': [1]}

    #parameters_randomforest = {'n_estimators': (25,50,100,150), 'criterion': ('entropy', 'gini'), 'max_depth': [10, 21, 42]}

    parameters_randomforest = {'n_estimators': [25], 'criterion': ['entropy'],
                               'max_depth': [10]}
    #parameters_nn = {'dropout_rate': (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 'neurons': (88, 128, 168, 208, 248),
    #                           'batch_size': [20, 40, 60, 80], 'epochs': [10, 50, 100]}

    parameters_nn = {'dropout_rate': [0.2], 'neurons': [88],
                     'batch_size': [20], 'epochs': [10]}

    parameters_list = [parameters_nn, parameters_decisiontree, parameters_svm, parameters_kneighbors, parameters_randomforest, parameters_logistic]

    model5 = LogisticRegression(random_state=0, multi_class='auto')
    model1 = DecisionTreeClassifier(random_state=0)
    model2 = SVC(tol=1e-3, random_state=0, gamma="scale",verbose=True)
    # model2 = SVC(kernel='linear')
    model3 = KNeighborsClassifier(metric='minkowski', algorithm= 'auto')
    model4 = RandomForestClassifier(random_state=1, n_jobs=2)
    model = KerasClassifier(build_fn=nn_model, input_dim=np.shape(X_train_)[1], verbose=0)
    model_list = [model, model1, model2, model3, model4, model5]
    model_name_list = ['NeuralNetwork', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest','LogisticRegression']

    for name, mod,parameter in zip(model_name_list,model_list, parameters_list):
        clf = GridSearchCV(mod, parameter, cv=5, scoring= 'balanced_accuracy')
        clf.fit(X_transform, y_train_)
        print(clf.cv_results_.keys())
        with open(name+'parameter.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(clf.cv_results_.keys())
            for i in range(len(clf.cv_results_['mean_fit_time'])):
                row = []
                for ele_key in clf.cv_results_.keys():
                    print(type(clf.cv_results_[ele_key]))
                    if(isinstance(clf.cv_results_[ele_key],np.ma.core.MaskedArray)):
                        row.append((clf.cv_results_[ele_key].data)[i])
                    else:
                        row.append(clf.cv_results_[ele_key][i])
                w.writerow(row)

trends = csvdata1()
trends_ratio_15,trends_ratio_30,trends_ratio_60,coef = calcTrends(trends)
raw_dict_list,raw_words_dict,target,length = csvdata()
y = np.array(target).ravel()
X = inputMatrix(raw_dict_list,raw_words_dict,trends_ratio_15,trends_ratio_30,trends_ratio_60,coef,length)
length = len(y)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)
# X is the feature set and y is the target
max_acc = 0
best_mod_list = []
tag = 0
X_train_, X_test_, y_train_, y_test_ = train_test_split(X,y, test_size=0.3, random_state=5)
grid_search(X_train_, X_test_, y_train_, y_test_)
#learning_rate(X_train_, y_train_)

for train_index, val_index in skf.split(X_train_,y_train_):
    #print("Train:", train_index, "Validation:", val_index)
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    print('Train: %s | test: %s' % (train_index, val_index))
    mod_list,train_acc_list, test_acc_list, test_neg_acc_list = train_predict(X_train, X_test, y_train, y_test,str(tag))
    tag = tag + 1
    if(max(test_neg_acc_list) > max_acc):
        best_mod_list = mod_list
        max_acc = max(test_neg_acc_list)

predict_exist(best_mod_list, X_test_, y_test_)
train_predict_exist(best_mod_list, X_train_, X_test_, y_train_, y_test_)
train_predict(X_train_, X_test_, y_train_, y_test_ ,'total')

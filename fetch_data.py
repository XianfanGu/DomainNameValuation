import csv
import numpy as np
import math
from keras.preprocessing.text import Tokenizer
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
pathname = 'dataset1_1.csv'
pathname1 = 'dataset2_1.csv'
pathname2 = 'dataset3_1.csv'
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

def train_predict(X_train, X_test, y_train, y_test):

    model_name_list = ['LogisticRegression', 'DecisionTree', 'SVM', 'KNeighbors', 'RandomForest']
    model = LogisticRegression(C=1e5, random_state=0, solver='liblinear', multi_class='auto', max_iter=5000)
    model1 = DecisionTreeClassifier(criterion='entropy', max_depth=21, random_state=0)
    model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=1e5, verbose=True)
    #model2 = SVC(kernel='linear')
    model3 = KNeighborsClassifier(n_neighbors=21, p=3, metric='minkowski')
    model4 = RandomForestClassifier(criterion='entropy', n_estimators=21, random_state=1,n_jobs=2)
    model_list = [model,model1,model2,model3,model4]
    for mod,name in zip(model_list,model_name_list):
        mod.fit(X_train, y_train)
        prediction_y = mod.predict(X_test)
        test = []
        predict = []
        j = 0
        for ele in y_test:
            if(ele == -1):
                test.append(-1)
                predict.append(prediction_y[j])
            j = j+1

        accuracy = accuracy_score(y_test, prediction_y)
        neg_acc = accuracy_score(test, predict)
        print(prediction_y)
        print(y_test)
        print("The accuracy of "+name+" model is: ",accuracy)
        print("The negative accuracy of "+name+" model is: ",neg_acc)

def batch_train_predict(batch_num,X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=10.0, verbose=True)  # 8.05 for SVM model
    model3 = KNeighborsClassifier(n_neighbors=21, p=3, metric='minkowski')  # 7.05 for  model
    model5 = DecisionTreeClassifier(criterion='entropy', max_depth=21, random_state=0)
    model4 = RandomForestClassifier(criterion='entropy', n_estimators=21, random_state=1,n_jobs=2)  # 5.2 for randomforest
    model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    train_batch_size = int(len(y_train)/(batch_num-1))
    train_rest_size = len(y_train)% train_batch_size
    test_batch_size = int(len(y_test)/(batch_num-1))
    test_rest_size = len(y_test)% test_batch_size
    print(train_batch_size,train_rest_size,test_batch_size,test_rest_size)
    for i in range(1,batch_num):
        print(i)
        X_train_input = X_train[(i-1)*train_batch_size:i*train_batch_size]
        y_train_input = y_train[(i-1)*train_batch_size:i*train_batch_size]
        X_test_input = X_test[(i - 1) * test_rest_size:i * test_rest_size]
        y_test_input = y_test[(i - 1) * test_rest_size:i * test_rest_size]
        model.fit(X_train_input, y_train_input)
        prediction_y = model.predict(X_test_input)
        accuracy = accuracy_score(y_test_input, prediction_y)
        print(prediction_y)
        print(y_test_input)
        print("The accuracy of LogisticRegression model is: ", accuracy)

    X_train_input = X_train[len(X_train)-train_rest_size:]
    y_train_input = y_train[len(y_train)-train_rest_size:]
    X_test_input = X_test[len(X_test)-test_rest_size:]
    y_test_input = y_test[len(y_test)-test_rest_size:]
    model.fit(X_train_input, y_train_input)
    prediction_y = model.predict(X_test_input)
    accuracy = accuracy_score(y_test_input, prediction_y)
    print(prediction_y)
    print(y_test_input)
    print("The accuracy of LogisticRegression model is: ", accuracy)

trends = csvdata1()
trends_ratio_15,trends_ratio_30,trends_ratio_60,coef = calcTrends(trends)
raw_dict_list,raw_words_dict,target,length = csvdata()
y = target
X = inputMatrix(raw_dict_list,raw_words_dict,trends_ratio_15,trends_ratio_30,trends_ratio_60,coef,length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
batch_num = 10
#batch_train_predict(batch_num,X_train, X_test, y_train, y_test)
train_predict(X_train, X_test, y_train, y_test)
#print(toOnehotkey(dict_list))

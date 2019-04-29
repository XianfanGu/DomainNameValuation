import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
pathname = 'statistic.csv'

def csvdata():
    try:
        with open(pathname, "r") as inp:
            words_dict = []
            dict_list = []
            trend = []
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
                trend_list = []
                words = str(row[3]).split(" ")
                words_dict.append(words)
                for ele in words:
                    dict_list.append(ele)
                for i in range(4,len(row)):
                    trend_list.append(int(row[i]))
                trend.append(trend_list)
            return dict_list,words_dict,trend

    except Exception as error:
        print(error)
        return None

def toOnehotkey(input_):
    keydict = {}
    tokenizer = Tokenizer(num_words=len(input_))
    tokenizer.fit_on_texts(input_)
    one_hot_results = tokenizer.texts_to_matrix(input_, mode='binary')
    for key,val in zip(input_,one_hot_results):
        keydict[key] = val
    return  keydict

def createTarget(trend, threshold = 0.01):
    targ = []
    for ele in trend:
        x = np.array([i for i in range(len(ele))]).reshape(-1, 1)
        y = np.array(ele).reshape(-1, 1)
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(x, y)
        targ.append((reg.coef_[0][0]))
    target = []
    for ele in targ:
        if (ele > threshold):
            target.append(1)
        else:
            target.append(0)
    return target

def inputMatrix(dict_list, words_dict):
    maxval = 0
    X_ = []
    hotkey_dict = toOnehotkey(dict_list)
    for ele in words_dict:
        if maxval < len(ele):
            maxval = len(ele)
    for ele1 in words_dict:
        length = len(ele1)
        arr = hotkey_dict[ele1[0]]
        for i in range(1, maxval):
            # print(i%length)
            arr = np.concatenate([arr, hotkey_dict[ele1[i % length]]])
        X_.append(arr)
    return X_
def train_predict(X_train, X_test, y_train, y_test):
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    prediction_y = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction_y)
    print("The accuracy of LogisticRegression model is: ",accuracy)

dict_list,words_dict,trend = csvdata()
y = createTarget(trend)
X = inputMatrix(dict_list,words_dict)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
train_predict(X_train, X_test, y_train, y_test)

#print(toOnehotkey(dict_list))

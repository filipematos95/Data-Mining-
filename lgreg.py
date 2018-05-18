from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix

train_file = 'train_1000000_preprocessed.csv'
test_file = 'test_1000000_preprocessed.csv'

columns_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,29,30,31,32,33]
columns_test = columns_train

train = pd.read_csv(train_file, usecols = columns_train,skiprows = [1,2])
test = pd.read_csv(test_file,usecols = columns_test,skiprows = [1,2])

NAN = train[train.isnull().any(axis=1)][['srch_id','clicked']]

train_x = train.drop(['booked','clicked'],axis = 1)
train_y = train['clicked']

imp_nan = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp_nan.fit(train_x)
train_x = imp_nan.transform(train_x)

test_x = test.drop(['booked','clicked'],axis = 1)
test_y = test['clicked']

test_x = imp_nan.transform(test_x)

plt.hist(train_y)
plt.show()

def compute(r):
    k = len(r)
    return ndcg_at_k(r,k)

def dcg_at_k(r,k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

# should be 5 columns searchid, clicked, booked, lr, prop_id  
def score(ex):
    book = 5*ex['booked']
    click = ex['clicked']
    ex['points'] = ex[['booked', 'clicked']].apply(max, axis = 1) 
    ex[['booked', 'clicked']].apply(np.max, axis = 1)
    ex = ex.sort_values(['srch_id', 'prob'],ascending=[True, False]) 
    ex['score'] = ex.groupby('srch_id').apply(lambda x: compute(x.points.values))
    #ex['score'] = ex.groupby('srch_id').apply(lambda x: ndcg_at_k(x.points.values))
    return ex

def Logistic_Regression(train_x,train_y,test_x,test_y,test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_x, train_y)
    pred = logisticRegr.predict(test_x)
    score = logisticRegr.score(test_x,test_y)
    prob = logisticRegr.predict_proba(test_x)

    print "LG -> Accuracy: " + str(score)
    print confusion_matrix(test_y, pred)

    result = test[['srch_id','prop_id','booked','clicked']]

    result = result.assign(prob = pd.Series(prob[:,1]) ,index = 'prob')

    return result

def Random_Forest(train_x,train_y,test_x,test_y):
    clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    score = clf.score(test_x,test_y)
    prob = clf.predict_proba(test_x)

    print "RF -> Accuracy: " + str(score)

    result = test[['srch_id','prop_id','clicked','booked']]

    result = result.assign(prob = pd.Series(prob[:,1]) ,index = 'prob')

    return result

result = Logistic_Regression(train_x,train_y,test_x,test_y,test)
temp = score(result).dropna()[['srch_id', 'score']]
print "LG -> NDCG Score: " + str(temp['score'].mean()) 
result = Random_Forest(train_x,train_y,test_x,test_y)
temp = score(result).dropna()[['srch_id', 'score']]
print "RF -> NDCG Score: " + str(temp['score'].mean()) 
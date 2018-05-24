from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix


train_file = 'train_1000000_preprocessed.csv'
test_file = '../Kaggle/expedia-personalized-sort/test_clean_preprocessed.csv'

train = pd.read_csv(train_file,skiprows = [1,2])
print train.columns.values 
train.drop(['position','gross_bookings_usd'],axis = 1)
test = pd.read_csv(test_file,skiprows = [1,2])
#test.drop(['position','gross_bookings_usd'],axis = 1)

#NAN = train[train.isnull().any(axis=1)][['srch_id','clicked']]

train_x = train.drop(['booked','clicked'],axis = 1)
train_y = train['clicked']

imp_nan = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp_nan.fit(train_x)
train_x = imp_nan.transform(train_x)

#test_x = test.drop(['booked','clicked'],axis = 1)
#test_y = test['clicked']

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

def write_to_kaggle(ex):
    ex = ex.sort_values(['srch_id', 'prob'],ascending=[True, False])
    return ex['srch_id', 'prop_id']

def Logistic_Regression(train_x,train_y,test_x,test_y):

    logisticRegr = LogisticRegression(n_jobs = -1,solver = 'lbfgs')
    logisticRegr.fit(train_x, train_y)
    pred = logisticRegr.predict(test_x)
    #score = logisticRegr.score(test_x,test_y)
    prob = logisticRegr.predict_proba(test_x)

    #print "LG -> Accuracy: " + str(score)
    #print confusion_matrix(test_y, pred)

    result = test[['srch_id','prop_id','booked','clicked']]

    result = result.assign(prob = pd.Series(prob[:,1]) ,index = 'prob')

    return result

def Random_Forest(train_x,train_y,test_x,test_y):
    clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    #score = clf.score(test_x,test_y)
    prob = clf.predict_proba(test_x)

    #print "RF -> Accuracy: " + str(score)

    result = test[['srch_id','prop_id','clicked','booked']]

    result = result.assign(prob = pd.Series(prob[:,1]) ,index = 'prob')

    return result
test_y = 1

result = Logistic_Regression(train_x,train_y,test_x,test_y)
result = write_to_kaggle(ex)
result.to_csv('Kaggle_LG.csv')
#temp = score(result).dropna()[['srch_id', 'score']]
#print "LG -> NDCG Score: " + str(temp['score'].mean()) 
result = Random_Forest(train_x,train_y,test_x,test_y)
result = write_to_kaggle(ex)
result.to_csv('Kaggle_RF.csv')
#temp = score(result).dropna()[['srch_id', 'score']]
#print "RF -> NDCG Score: " + str(temp['score'].mean()) 

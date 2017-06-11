import datetime
import numpy as np
import pandas as pd
import sklearn
import quandl
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA


def create_series(symbol, start_date, end_date, lags=5) :

  ts = quandl.get("WIKI/%s"%(symbol), start_date = start_date, end_date = end_date)
  tslag = pd.DataFrame(index=ts.index)
  tslag["Today"] = ts["Adj. Close"]
  tslag["Volume"] = ts["Volume"]

  for i in xrange(0,lags):
    tslag["Lag%s" % str(i+1)] = ts["Adj. Close"].shift(i+1)

  tsret = pd.DataFrame(index=tslag.index)
  tsret["Volume"] = tslag["Volume"]
  tsret["Today"] = tslag["Today"].pct_change()*100.0

  for i,x in enumerate(tsret["Today"]):
    if (abs(x) < 0.0001):
      tsret["Today"][i] = 0.0001

  for i in xrange(0,lags):
    tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

  tsret["Direction"] = np.sign(tsret["Today"])
  tsret = tsret[tsret.index >= start_date]

  return tsret


def fit_model(name, model, X_train, y_train, X_test, pred):

  model.fit(X_train, y_train)
  pred[name] = model.predict(X_test)

  pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
  hit_rate = np.mean(pred["%s_Correct" % name])
  print "%s : %.3f" % (name,hit_rate)


if __name__ == "__main__":
  
  snpret = create_series(sys.argv[1], datetime.datetime(1996,1,1), datetime.datetime(2016,12,31), lags=5)

  X = snpret[["Lag1","Lag2"]]
  y = snpret["Direction"]

  start_test = datetime.datetime(2016,1,1)

  X_train = X[X.index < start_test]
  X_test = X[X.index >= start_test]
  y_train = y[y.index < start_test]
  y_test = y[y.index >= start_test]

  x_train_nan = X_train.isnull().sum().sum()
  y_train_nan = y_train.isnull().sum().sum()

  if x_train_nan > y_train_nan :
    X_train = X_train[x_train_nan:]
    y_train = y_train[x_train_nan:]
  else :
    X_train = X_train[y_train_nan:]
    y_train = y_train[y_train_nan:]

  
  pred = pd.DataFrame(index= y_test.index)
  pred["Actual"] = y_test

  print "Hit rates:"
  models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
  for m in models:
    fit_model(m[0], m[1], X_train, y_train, X_test, pred)

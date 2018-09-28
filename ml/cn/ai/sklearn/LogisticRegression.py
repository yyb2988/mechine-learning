from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

if __name__ == '__main__':

    #print(iris.data)
    #print(type(iris.target))
    #1.模型训练和保存
    iris = datasets.load_iris()
    lr = LogisticRegression ()
    lr.fit(iris.data,iris.target);
    joblib.dump(lr,"LogisticRegression.model");

    #模型使用
    model = joblib.load("LogisticRegression.model")
    result = model.predict([[5,3,5,2.5],[133,8,9,1.5]])
    print(result[0])
    print(result[1])





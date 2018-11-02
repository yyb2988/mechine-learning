import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# https://www.cnblogs.com/amberdata/p/7203632.html
# https://blog.csdn.net/zjuPeco/article/details/77371645?locationNum=7&fps=1

if __name__ == '__main__':
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    df = pd.read_csv(url, header=None)
    df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                  'Alcalinity of ash', 'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                  'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    classes = np.unique(df['Class label'])
    print(classes)
    feat_labels = df.columns[1:]
    print(df.info())
    y = (df.iloc[:, 0:1]).values
    x = (df.iloc[:, 1:]).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    forest = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=-1, verbose=5, max_features="log2")

    forest.fit(x_train, y_train.reshape(-1))
    y_test_one_dim = y_test.reshape(-1)

    pre_test_y = forest.predict(x_test)

    acc = 0
    for index, pre_y in enumerate(pre_test_y):
        if pre_y == y_test_one_dim[index]:
            # print(pre_y, y_test_one_dim[index])
            acc += 1
    print('acc:'+str(acc)+',total:'+str(len(pre_test_y)))

    importances = forest.feature_importances_
    importances_2 = np.argsort(importances)[::-1]
    for _, clazz in enumerate(importances_2):
        print(clazz, feat_labels[clazz], importances[clazz])

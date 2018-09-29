import numpy as np
import matplotlib.pyplot as pt
import pandas as pd


from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()

clt = DecisionTreeClassifier() ;

x_train = data[0:21000,1:]
train_label = data[0:21000,0]

clt.fit(x_train,train_label)

x_test = data[21000:,1:]
actual_label = data[21000:,0]

d = x_test[9]
d.shape = (28,28)
pt.imshow(255-d,cmap='gray')
print(clt.predict( [x_test[9]] ))
pt.show()

p = clt.predict(x_test)

count = 0
for i in range(0,21000):
    count+=1 if p[i] == actual_label[i] else 0
print("Accuracy=" , (count/21000)*100)



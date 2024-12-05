# Uses KNN

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import os

csv_file = "datasets/tele_customers.csv"
df = pd.read_csv(csv_file)

# df.hist(column='income', bins=50)
# plt.show()
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
Y = df[['custcat']].values

# Normlize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Ks = 11
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
print(mean_acc)


for n in range(1, Ks):
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, prediction)

    std_acc[n-1]=np.std(prediction==y_test)/np.sqrt(prediction.shape[0])


os.system('clear')
print(mean_acc)
print(std_acc)

plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
# plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("Best Accuracy:", mean_acc.max(), "With K:", mean_acc.argmax()+1)
# Prépartion des données

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

#On charge les données depuis le fichier Data.csv
df = pd.read_csv('Social_Network_Ads.csv')

# On recupere Les valeurs sans les titres
x = df.iloc[:, [2, 3]].values

#On recupere les valeurs du derniers comonne
y = df.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

# Regression logistique

classifier = LogisticRegression(random_state=0)

classifier.fit(x_train, y_train)

# Prediction sur le jeu de données test

y_pred = classifier.predict(x_test)

#Matrice de confusion

cm = confusion_matrix(y_test, y_pred)

#Visualiser  les resultats

x_set, y_set = x_train, y_train

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(x1.min(), x1.max())

plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
              c = ListedColormap(('red', 'blue'))(i),label=j)
  
plt.title('Classifier training set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()








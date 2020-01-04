# Prépartion des données

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#On charge les données depuis le fichier Data.csv
df = pd.read_csv('Data.csv')

# On recupere Les valeurs sans les titres
x = df.iloc[:, :-1].values

#On recupere les valeurs du derniers comonne
y = df.iloc[:, -1].values

# on cree un objet imputer pour gerer les valeur manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(x[:, 1:3])
#On transforme les valeurs manquantes en leur assignant une valeur selon l'objet imputer
x[:,1:3] = imputer.transform(x[:, 1:3])

labelencoder_x = LabelEncoder()

x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])

x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()

y= labelencoder_x.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)



# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import random
import warnings

from matplotlib import style
from sklearn import ensemble
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, GridSearchCV
from datetime import datetime
from pylab import rcParams

style.use('fivethirtyeight')
# %matplotlib inline
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({'font.size': 14})

def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
    print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
    print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))

    plt.figure(figsize=(18,10))

    plt.subplot(121)
    sns.scatterplot(x=train_pred_values, y=train_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Train sample prediction')

    plt.subplot(122)
    sns.scatterplot(x=test_pred_values, y=test_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Test sample prediction')

    plt.show()

TEST_DATASET_PATH = 'test.csv'
TRAIN_DATASET_PATH = 'train.csv'

test_data = pd.read_csv(TEST_DATASET_PATH)
train_data = pd.read_csv(TRAIN_DATASET_PATH)

"""# Описание датасета

- **Id** - идентификационный номер квартиры
- **DistrictId** - идентификационный номер района
- **Rooms** - количество комнат
- **Square** - площадь
- **LifeSquare** - жилая площадь
- **KitchenSquare** - площадь кухни
- **Floor** - этаж
- **HouseFloor** - количество этажей в доме
- **HouseYear** - год постройки дома
- **Ecology_1, Ecology_2, Ecology_3** - экологические показатели местности
- **Social_1, Social_2, Social_3** - социальные показатели местности
- **Healthcare_1, Healthcare_2** - показатели местности, связанные с охраной здоровья
- **Shops_1, Shops_2** - показатели, связанные с наличием магазинов, торговых центров
- **Price** - цена квартиры

"""

train_data.tail()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='Rooms', y='Price', data=train_data)
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Distribution of Price by Rooms')
plt.show()

"""Исключение малозначимых столбцов, содержащих количественные признаки"""

train_data.dtypes

df_num_features = train_data.select_dtypes(include='float64')

num_features = pd.DataFrame(df_num_features)
num_features.drop('Ecology_1', axis=1, inplace=True)
num_features.drop('Healthcare_1', axis=1, inplace=True)
num_features.hist(figsize=(10, 8), bins=20, grid=False)

"""Взаимосвязь признаков"""

corr = num_features.corr()
plt.figure(figsize=(8, 8))
mask = np.zeros_like(corr, dtype=bool)  # Изменено с np.bool на bool
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale=1.4)
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, cmap='GnBu')
plt.title('Relationship matrix')
plt.show()

test_data.info()

"""Заполнение пропусков"""

train_data = train_data.sort_values('Price')
test_data = test_data.sort_values('DistrictId')
train_data = train_data.fillna(method='pad')
test_data = test_data.fillna(method='pad')

"""Проверка данных"""

rcParams['figure.figsize'] = 12, 6  # изменение размера графиков

plt.scatter(train_data.Price, train_data.Square)

"""Формирование и обучение модели. Предсказание цен"""

# формирование модели

X_train = train_data.drop('Price', axis=1) # 1
y_train = train_data['Price'] # 1

X_test = test_data # 2

pr = pd.DataFrame()
pr['Id'] = X_test['Id'].copy() # 2.0

del_list = ["Id", "DistrictId", "LifeSquare", "Healthcare_1", "Ecology_2", "Ecology_3", "Shops_2"]
X_train.drop(del_list, axis=1, inplace=True) # 1
X_test.drop(del_list, axis=1, inplace=True) # 2

# обучение модели
model = RandomForestRegressor(n_estimators=1000, max_depth=16, random_state=42, max_features=7)
model.fit(X_train, y_train) # 1

y_pred = model.predict(X_test) # 2

# предсказание цен
pr['Price'] = y_pred # 2.0
pr.to_csv('predictions.csv', index=False) # 2.0
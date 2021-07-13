import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from cymem.cymem import Pool
from numpy import mean
from numpy import std
from catboost import CatBoostClassifier
from catboost import Pool
from catboost import MetricVisualizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from pickle import dump
import matplotlib
from pickle import load

data = pd.read_csv('DataT3RR - Copy.csv')
print('Size of weather data frame is :',data.shape)
data.info()
data.count().sort_values()

data = data.drop(columns=['Stasiun','Tanggal'], axis=1)
data = data.dropna(how='any')
print(data.shape)

cor = data.corr(method='kendall')
sb.heatmap(cor, square = True , annot=True)
# plt.show()

X = data.drop(columns=['Tn','Tavg','Besok_hujan'], axis=1)
y = data['Besok_hujan']

model = CatBoostClassifier(iterations=500, custom_loss=['AUC', 'Accuracy'], train_dir='auto_rate')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
validation_pool = Pool(data=X_test, label=y_test)

model.fit(X, y, eval_set=validation_pool,verbose=False, plot=True)

print('Accuracy : ', model.score(X, y))
dump(model, open('model2.pkl', 'wb'))
print('Model Saved..!!')

# row = [[32.7,	44.2,	1]]
# yhat = model.predict(row)
# print('Prediction: %d' % yhat[0])
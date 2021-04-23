import pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy

from sklearn.model_selection import train_test_split
from sklearn import metrics

import keras
from keras import layers
from keras.models import Sequential

dataset = pandas.read_csv("dataset.csv")

dataset = dataset.drop(['filename'], axis=1)
dataset = dataset.drop(dataset.columns[0], axis=1)

target = dataset.iloc[:,-1]
encoder = LabelEncoder()
target = encoder.fit_transform(target)

data = dataset.iloc[:,:-1]
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype=float))


print(target)
print(data)

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.2)

machine = Sequential()
machine.add(layers.Dense(256, activation='relu', input_shape=(data_training.shape[1],)))
machine.add(layers.Dense(128, activation='relu'))
machine.add(layers.Dense(64, activation='relu'))
machine.add(layers.Dense(10, activation='softmax'))
machine.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

machine.fit(data_training, target_training, epochs=100, batch_size=128)

new_target = numpy.argmax(machine.predict(data_test),axis=-1)
print(new_target)
print("Accuracy Score (Deep Learning): " + str(metrics.accuracy_score(new_target, target_test)))



# 





# Compare to Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest_machine = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=11)
random_forest_machine.fit(data_training, target_training)
new_target = random_forest_machine.predict(data_test)
print("Accuracy score(Random Forest): ", metrics.accuracy_score(target_test,new_target))












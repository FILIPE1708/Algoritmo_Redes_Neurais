import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

csv = pd.read_csv('dados.csv', sep=',')
csv = csv.drop(columns=['lote'])
labelEnc = LabelEncoder()
csv['fruta'] = labelEnc.fit_transform(csv['fruta'])
dados = csv.values
atributos = dados[:,1:]
classificadores = dados[:,0]

modelo = Sequential()
modelo.add(Dense(units=5, activation='relu'))
modelo.add(Dense(units=1, activation='sigmoid'))
modelo.compile(optimizer='adam',  loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
modelo.fit(atributos, classificadores, batch_size=10, epochs=100)
modelo.save('modelo.h5')
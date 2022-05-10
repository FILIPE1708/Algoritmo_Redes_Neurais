from keras.models import load_model
import numpy as np

modelo = load_model('modelo.h5')

novosDados = np.array([
    [3.1, 122],
    [4.1, 146],
    [2.2, 86]
])

resultado = modelo.predict(novosDados)
for i in range(len(resultado)):
    if (resultado[i] < 0.5):
        print('Laranja')

    elif (resultado[i] >= 0.5):
        print('Limão')
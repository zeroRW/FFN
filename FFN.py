import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Generar datos sintéticos para el ejemplo
X = np.linspace(-1, 1, 100)
y = X**2 + np.random.normal(0, 0.2, 100)

# Crear modelo
model = Sequential()

# Agregar capas al modelo
model.add(Dense(units=10, activation='relu', input_dim=1))
model.add(Dense(units=1, activation='linear'))

# Compilar modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar modelo
history = model.fit(X, y, epochs=1000, verbose=0)

# Hacer predicciones con el modelo
y_pred = model.predict(X)

# Graficar los resultados
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Resultado de la FFN para la función y')
plt.show()
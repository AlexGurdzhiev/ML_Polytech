
# import numpy as np


# def neuralNetwork(inp, weights):
#     prediction_h = inp.dot(weights[0])
#     prediction_out = prediction_h.dot(weights[1])
#     return prediction_out

# inp = np.array([23, 45])
# weight_h_1 = [0.4, 0.1]
# weight_h_2 = [0.3, 0.2]

# weight_out_1 = [0.4, 0.1]
# weight_out_2 = [0.3, 0.1]

# weights_h = np.array([weight_h_1, weight_h_2]).T #транспонируем весовые матрицы
# weights_out = np.array([weight_out_1, weight_out_2]).T #транспонируем весовые матрицы

# weights = [weights_h, weights_out]
# print(neuralNetwork(inp, weights))



import numpy as np


def neuralNetwork(inp, weights):
    # Проход через все слои
    prediction_h1 = inp.dot(weights[0])  # Первый скрытый слой
    prediction_h2 = prediction_h1.dot(weights[1])  # Второй скрытый слой
    prediction_out = prediction_h2.dot(weights[2])  # Выходной слой
    return prediction_out


# Входные данные
inp = np.array([23, 45])

# Случайная генерация весов
np.random.seed(42)  # Для воспроизводимости результатов

# Генерируем случайные веса для трех скрытых слоев
# Форма весов: (входной_размер, выходной_размер)
weights_h1 = np.random.randn(2, 3)  # Вход: 2, выход: 3 нейрона
weights_h2 = np.random.randn(3, 2)  # Вход: 3, выход: 2 нейрона
weights_out = np.random.randn(2, 1)  # Вход: 2, выход: 1 нейрон

# Альтернативный вариант с фиксированными весами (как в задании):
# weights_h1 = np.array([[0.4, 0.1, 0.6], [0.3, 0.2, 0.2]]).T  # (2,3)
# weights_h2 = np.array([[0.4, 0.1], [0.3, 0.1], [0.6, 0.2]]).T  # (3,2)
# weights_out = np.array([[0.4], [0.3]])  # (2,1)

weights = [weights_h1, weights_h2, weights_out]

# Выполняем предсказание
result = neuralNetwork(inp, weights)

print("Входные данные:", inp)
print("\nВеса первого скрытого слоя (2x3):")
print(weights[0])
print("\nВеса второго скрытого слоя (3x2):")
print(weights[1])
print("\nВеса выходного слоя (2x1):")
print(weights[2])
print("\nПредсказание сети:", result)
print("Форма выхода:", result.shape)
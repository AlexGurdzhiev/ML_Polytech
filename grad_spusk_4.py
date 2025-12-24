# ----------------------------Градиентный спуск с несколькими входами и выходами
import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

# Данные
inp = np.array([8.5, 0.65, 1.2])  # 3 входа
weights = np.array([[0.1, 0.1, -0.3],  # 3x3 матрица весов
                    [0.1, 0.2, 0.0],
                    [0.0, 1.3, 0.1]])
true_values = np.array([0.5, 1.0, -0.1])  # 3 выхода
learning_rate = 0.001

print("Обучение с несколькими входами и выходами:")
for i in range(500):
    # Прямой проход
    predictions = neural_network(inp, weights)
    error = np.sum((true_values - predictions) ** 2)
    
    if i % 100 == 0:
        print(f"Эпоха {i}: Error={error:.8f}")
    
    # Обновление весов
    for i in range(weights.shape[0]):  # строки
        for j in range(weights.shape[1]):  # столбцы
            gradient = 2 * (predictions[j] - true_values[j]) * inp[i]
            weights[i][j] = weights[i][j] - learning_rate * gradient
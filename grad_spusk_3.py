# -----------------------------Градиентный спуск с несколькими выходами

import numpy as np

def neural_network(inp, weights):
    return inp * weights  # Поэлементное умножение

def get_error(true_vals, preds):
    return np.sum((true_vals - preds) ** 2)

# Данные
inp = 0.5  # Один вход
weights = np.array([0.3, 0.2, 0.9])  # 3 веса для 3 выходов
true_values = np.array([0.4, 0.3, 0.9])  # 3 целевых значения
learning_rate = 0.01

print("Обучение с несколькими выходами:")
for i in range(100):
    # Прямой проход
    predictions = neural_network(inp, weights)
    error = get_error(true_values, predictions)
    
    if i % 20 == 0:
        print(f"Эпоха {i}: Predictions={predictions}, Error={error:.8f}")
    
    # Градиент для каждого веса
    for j in range(len(weights)):
        gradient = 2 * (predictions[j] - true_values[j]) * inp
        weights[j] = weights[j] - learning_rate * gradient
import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

def get_error(true_val, pred):
    return (true_val - pred) ** 2

# Данные
inp = np.array([8.5, 0.65, 1.2])  # 3 признака
weights = np.array([0.1, 0.2, -0.1])  # 3 веса
true_value = 0.5
learning_rate = 0.01

print("Обучение с несколькими входами:")
for i in range(100):
    # Прямой проход
    prediction = neural_network(inp, weights)
    error = get_error(true_value, prediction)
    
    if i % 20 == 0:
        print(f"Эпоха {i}: Prediction={prediction:.5f}, Error={error:.8f}")
    
    # Градиент для каждого веса
    for j in range(len(weights)):
        gradient = 2 * (prediction - true_value) * inp[j]
        weights[j] = weights[j] - learning_rate * gradient
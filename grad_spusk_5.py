#----------------------------Обучение на нескольких наборах данных

def neural_network(inp, weights):
    return inp.dot(weights)

def get_error_mse(true_vals, preds):
    # Среднеквадратичная ошибка (MSE)
    return np.mean((true_vals - preds) ** 2)

def get_error_rmse(true_vals, preds):
    # RMSE (корень из MSE)
    return np.sqrt(np.mean((true_vals - preds) ** 2))

# Несколько наборов данных
inputs = np.array([
    [8.5, 0.65, 1.2],
    [9.5, 0.8, 1.3],
    [9.9, 0.8, 0.5],
    [9.0, 0.9, 1.0]
])

true_outputs = np.array([
    [0.5, 1.0, -0.1],
    [0.6, 0.9, 0.0],
    [0.8, 0.7, -0.2],
    [0.7, 0.8, -0.1]
])

weights = np.array([[0.1, 0.1, -0.3],
                    [0.1, 0.2, 0.0],
                    [0.0, 1.3, 0.1]])
learning_rate = 0.001

print("Обучение на нескольких наборах:")
for epoch in range(100):  # Было 500
    total_error = 0
    
    for i in range(len(inputs)):
        # Прямой проход
        prediction = neural_network(inputs[i], weights)
        error = get_error_mse(true_outputs[i], prediction)
        total_error += error
        
        # Обратное распространение
        for row in range(weights.shape[0]):
            for col in range(weights.shape[1]):
                grad = 2 * (prediction[col] - true_outputs[i][col]) * inputs[i][row]
                weights[row][col] -= learning_rate * grad
    
    if epoch % 20 == 0:
        avg_error = total_error / len(inputs)
        print(f"Эпоха {epoch}: Средняя ошибка={avg_error:.8f}")
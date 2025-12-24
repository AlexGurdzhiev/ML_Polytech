#----------Решение проблемы с расхождением
#1)
import numpy as np

# Простейшая нейросеть: вход * вес
def neural_networks(inp, weight):
    return inp * weight

# Функция ошибки (квадратичная)
def get_error(true_val, pred):
    return (true_val - pred) ** 2

# Инициализация
inp = 0.9
true_value = 0.2
weight = 0.5
learning_rate = 0.1  # Скорость обучения

print("Обучение нейросети:")
for i in range(10):
    # Прямой проход
    prediction = neural_networks(inp, weight)
    error = get_error(true_value, prediction)
    
    print(f"Эпоха {i}: Prediction={prediction:.5f}, Weight={weight:.5f}, Error={error:.8f}")
    
    # Градиентный спуск
    gradient = 2 * (prediction - true_value) * inp
    weight = weight - learning_rate * gradient

print(f"\nИдеальный вес: {true_value/inp:.5f}")
print(f"Финальный вес: {weight:.5f}")



#2)
def test_learning_rates():
    inp = 0.9
    true_value = 0.2
    weight = 0.5
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    
    for lr in learning_rates:
        w = weight
        print(f"\nLearning_rate = {lr}")
        
        for i in range(20):
            pred = inp * w
            grad = 2 * (pred - true_value) * inp
            w = w - lr * grad
            
            if i % 5 == 0:
                print(f"  Эпоха {i}: Weight={w:.5f}, Pred={pred:.5f}")
    
# Результаты:
# lr=0.0001: обучается очень медленно
# lr=0.01: обучается стабильно (оптимально)
# lr=0.5: может "проскочить" минимум
# lr=1.0: расходится (вес прыгает)

#3)
def test_iterations():
    inp = 0.9
    true_value = 0.2
    weight = 0.5
    lr = 0.1
    
    for epochs in [5, 10, 20, 50, 100]:
        w = weight
        for i in range(epochs):
            pred = inp * w
            grad = 2 * (pred - true_value) * inp
            w = w - lr * grad
        
        final_pred = inp * w
        error = abs(true_value - final_pred)
        print(f"Эпох: {epochs:3d} → Вес: {w:.6f} → Ошибка: {error:.8f}")
    
# Обычно хватает 20-30 эпох для хорошей точности
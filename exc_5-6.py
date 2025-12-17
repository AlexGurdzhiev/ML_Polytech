# Базовый код для нескольких выходов
print("5-6. Нейросеть с несколькими выходами:")

def multi_output_neural_network(inputs, weights_list):
    outputs = [0] * len(weights_list)
    for i, weights in enumerate(weights_list):
        for inp, w in zip(inputs, weights):
            outputs[i] += inp * w
    return outputs

# Исходные данные
inputs = [8.5, 0.65, 1.2]
weights = [
    [0.1, 0.2, -0.1],  # Веса для первого выхода
    [0.5, 0.3, -0.2],  # Веса для второго выхода
    [0.0, -0.3, 0.1]   # Веса для третьего выхода
]

# 5. Поиск весов для выхода > 0.5
print("5. Поиск весов для выхода > 0.5:")

def find_weights_for_threshold(inputs, target=0.5, step=0.01):
    weights_found = []
    for i in range(len(weights)):
        w = list(weights[i])  # Копируем исходные веса
        output = 0
        while output <= target:
            # Увеличиваем все веса на шаг
            w = [x + step for x in w]
            output = sum(inp * w_j for inp, w_j in zip(inputs, w))
        weights_found.append(w)
        print(f"Выход {i}: веса {[round(x, 3) for x in w]}, результат: {output:.3f}")
    return weights_found

found_weights = find_weights_for_threshold(inputs)

# 6. Цикл для поиска весов с остановкой
print("\n6. Поиск весов с остановкой:")
def find_weights_with_stop(inputs, initial_weights, target=0.5, step=0.01):
    current_weights = [list(w) for w in initial_weights]
    outputs = multi_output_neural_network(inputs, current_weights)
    completed = [False, False, False]
    iterations = 0
    max_iterations = 1000
    
    while not all(completed) and iterations < max_iterations:
        iterations += 1
        for i in range(len(current_weights)):
            if not completed[i]:
                # Увеличиваем веса для этого выхода
                for j in range(len(current_weights[i])):
                    current_weights[i][j] += step
                
                # Проверяем выход
                outputs = multi_output_neural_network(inputs, current_weights)
                if outputs[i] > target:
                    completed[i] = True
        
        if iterations % 100 == 0:
            print(f"Итерация {iterations}: выходы = {[round(x, 3) for x in outputs]}")
    
    print(f"\nНайдены веса после {iterations} итераций:")
    for i, w in enumerate(current_weights):
        print(f"Выход {i}: веса {[round(x, 3) for x in w]}, результат: {outputs[i]:.3f}")
    return current_weights

initial_weights = [
    [0.1, 0.2, -0.1],
    [0.1, 0.1, 0.0],
    [0.0, 0.0, 0.0]
]
final_weights = find_weights_with_stop(inputs, initial_weights)
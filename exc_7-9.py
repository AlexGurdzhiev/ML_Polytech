print("\n7-9. Нейросеть с несколькими входами и выходами:")

# 7. Добавление нового набора весов
def neural_network_multi(inputs, weights):
    outputs = [0] * len(weights)
    
    for i in range(len(outputs)):
        for j in range(len(inputs)):
            outputs[i] += inputs[j] * weights[i][j]
    
    return outputs

# Исходные данные
inputs_multi = [8.5, 9.5, 9.9, 9.0]
weights_multi = [
    [0.1, 0.2, -0.1, 0.5],   # weights_1
    [-0.1, 0.1, 0.9, 0.1],   # weights_2
    [0.3, 0.1, 0.0, 0.2]     # weights_3
]

# Добавляем weights_4
weights_4 = [0.4, 0.2, 0.1, -0.2]
weights_multi_with_4 = weights_multi + [weights_4]

print("7. С добавленным weights_4:")
results_without_4 = neural_network_multi(inputs_multi, weights_multi)
results_with_4 = neural_network_multi(inputs_multi, weights_multi_with_4)

print(f"Без weights_4: {[round(x, 3) for x in results_without_4]}")
print(f"С weights_4: {[round(x, 3) for x in results_with_4]}")
print("Добавление нового набора весов добавило новый выход в результат.\n")

# 8. Сделать выходы равными (метод проб и ошибок)
print("8. Делаем выходы равными (проб и ошибки):")

def make_outputs_equal_manual(inputs, weights):
    # Копируем веса
    w = [list(row) for row in weights]
    
    # Настраиваем веса вручную
    w[0] = [0.2, 0.15, 0.1, 0.25]  # Подбираем значения
    w[1] = [0.2, 0.15, 0.1, 0.25]  # Делаем одинаковыми
    
    outputs = neural_network_multi(inputs, w)
    print(f"Веса: {[[round(x, 3) for x in row] for row in w]}")
    print(f"Выходы: {[round(x, 3) for x in outputs]}")
    print(f"Разница между выходами: {abs(outputs[0] - outputs[1]):.3f}")
    
    return w

equal_weights = make_outputs_equal_manual(inputs_multi, weights_multi)

# 9. Автоматический поиск равных выходов
print("\n9. Автоматический поиск равных выходов:")

def find_equal_outputs(inputs, initial_weights, learning_rate=0.001, tolerance=0.01):
    w = [list(row) for row in initial_weights]
    iterations = 0
    max_iterations = 10000
    
    while iterations < max_iterations:
        outputs = neural_network_multi(inputs, w)
        diff = outputs[0] - outputs[1]
        
        if abs(diff) < tolerance:
            break
        
        # Корректируем веса
        if diff > 0:  # Первый выход больше
            for j in range(len(w[0])):
                w[0][j] -= learning_rate * inputs[j]
                w[1][j] += learning_rate * inputs[j]
        else:  # Второй выход больше
            for j in range(len(w[0])):
                w[0][j] += learning_rate * inputs[j]
                w[1][j] -= learning_rate * inputs[j]
        
        iterations += 1
    
    final_outputs = neural_network_multi(inputs, w)
    print(f"Найдено за {iterations} итераций:")
    print(f"Веса: {[[round(x, 3) for x in row] for row in w]}")
    print(f"Выходы: {[round(x, 3) for x in final_outputs]}")
    print(f"Разница: {abs(final_outputs[0] - final_outputs[1]):.5f}")
    
    return w

# Используем только первые 2 выхода для поиска равенства
initial_for_equal = [weights_multi[0], weights_multi[1]]
found_equal = find_equal_outputs(inputs_multi, initial_for_equal)
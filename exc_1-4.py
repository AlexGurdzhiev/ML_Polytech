# Базовый пример для задания 1-4
def neural_network(inputs, weights):
    prediction = 0
    for inp, w in zip(inputs, weights):
        prediction += inp * w
    return prediction

# 1. Измените входные данные и вес
print("1. Изменение входных данных и веса:")
inputs_1 = [8.5, 0.65, 1.2]  # Было [8.5, 0.65, 1.2]
weights_1 = [0.1, 0.2, -0.1]  # Было [0.1, 0.2, 0]
result_1 = neural_network(inputs_1, weights_1)
print(f"Входы: {inputs_1}, Веса: {weights_1}")
print(f"Результат: {result_1}")
print("Вывод изменился потому что веса умножаются на входы и суммируются.")
print(f"С отрицательным весом результат уменьшается.\n")

# 2. Цикл для списка входных данных
print("2. Вычисление для списка входов:")
inputs_list = [150, 160, 170, 180, 190]
weights_2 = [0.1, 0.2, 0]
for inp in inputs_list:
    # Создаем фиктивный список входов для одной переменной
    result = neural_network([inp, 1, 1], weights_2)
    print(f"Вход: {inp}, Выход: {result}")

# 3. Модификация с bias
print("\n3. Нейросеть с bias:")
def neural_network_with_bias(inp, weight, bias):
    return inp * weight + bias

# Пример
inputs_3 = [8.5, 0.65, 1.2]
weights_3 = [0.1, 0.2, -0.1]
bias = 0.5

for i, (inp, w) in enumerate(zip(inputs_3, weights_3)):
    result = neural_network_with_bias(inp, w, bias)
    print(f"Вход {i}: inp={inp}, weight={w}, bias={bias}, результат={result}")

print("Bias добавляет постоянное смещение, смещая выход вверх.\n")

# 4. Возврат промежуточных значений
print("4. Промежуточные значения:")
def neural_network_detailed(inputs, weights):
    intermediate = []
    prediction = 0
    for inp, w in zip(inputs, weights):
        intermediate_value = inp * w
        intermediate.append(intermediate_value)
        prediction += intermediate_value
    return prediction, intermediate

pred, intermediates = neural_network_detailed(inputs_1, weights_1)
print(f"Входы: {inputs_1}")
print(f"Веса: {weights_1}")
print(f"Промежуточные значения: {intermediates}")
print(f"Итоговый результат: {pred}\n")
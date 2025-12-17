print("\n10. Нейросеть со скрытым слоем:")

def neural_network_hidden(inputs, weights_ih, weights_ho):
    # Скрытый слой
    hidden = [0] * len(weights_ih[0])
    for j in range(len(hidden)):
        for i in range(len(inputs)):
            hidden[j] += inputs[i] * weights_ih[i][j]
    
    # Выходной слой
    outputs = [0] * len(weights_ho[0])
    for j in range(len(outputs)):
        for i in range(len(hidden)):
            outputs[j] += hidden[i] * weights_ho[i][j]
    
    return outputs, hidden

# Исходные данные
inputs_h = [8.5, 0.65, 1.2]

# Веса от входа к скрытому слою (3 входа, 4 скрытых нейрона)
weights_ih = [
    [0.1, 0.2, -0.1, 0.5],
    [0.3, -0.2, 0.4, 0.1],
    [-0.3, 0.4, 0.1, -0.2]
]

# Веса от скрытого слоя к выходу (4 скрытых нейрона, 2 выхода)
weights_ho = [
    [0.2, 0.1],
    [0.3, -0.1],
    [-0.2, 0.4],
    [0.1, 0.2]
]

# 10. Делаем prediction_h > 5
print("10. Делаем выходы скрытого слоя > 5:")

def increase_hidden_output(inputs, weights_ih, target=5.0, step=0.1):
    w_ih = [list(row) for row in weights_ih]
    hidden = []
    
    # Увеличиваем веса пока все скрытые нейроны не превысят target
    while True:
        # Вычисляем скрытый слой
        hidden = [0] * len(w_ih[0])
        for j in range(len(hidden)):
            for i in range(len(inputs)):
                hidden[j] += inputs[i] * w_ih[i][j]
        
        # Проверяем условие
        all_above_target = all(h > target for h in hidden)
        if all_above_target:
            break
        
        # Увеличиваем веса
        for i in range(len(w_ih)):
            for j in range(len(w_ih[i])):
                w_ih[i][j] += step
    
    outputs, final_hidden = neural_network_hidden(inputs, w_ih, weights_ho)
    
    print(f"Полученные веса вход->скрытый:")
    for i, row in enumerate(w_ih):
        print(f"  Нейрон {i}: {[round(x, 3) for x in row]}")
    
    print(f"\nСкрытый слой: {[round(x, 3) for x in final_hidden]}")
    print(f"Выходной слой: {[round(x, 3) for x in outputs]}")
    
    return w_ih

increased_weights = increase_hidden_output(inputs_h, weights_ih, target=5.0)
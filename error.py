

# import numpy as np

# def neural_networks(inp, weights):
#     return inp.dot(weights)

# def get_error(true_prediction, prediction):

#     return (true_prediction - prediction) ** 2

# prediction = neural_networks(np.array([150,40]), [0.2, 0.3]) #преобразование второго массива к объекту np произойдет автоматически
# print(prediction)

# #но предположим, что ожидаемый правильный ответ был 50
# true_prediction = 50
# print(get_error(true_prediction, neural_networks(inp, weights)))

import numpy as np
def neural_network(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction)**2

inp = 0.9
weight = 0.2
true_prediction = 0.2
for i in range (10):
    prediction = neural_network(inp, weight)
    error = get_error(true_prediction, prediction)
    print("prediction: %.10f, weight: %.5f, error: %.20f" %(prediction, weight, error))
    delta = (prediction - true_prediction) * inp

    weight = weight - delta



# import numpy as np

# def neural_networks(inp, weight):
#     return inp * weight

# def get_error(true_prediction, prediction):
#     return (true_prediction - prediction) ** 2

# # Инициализация
# inp = 30
# true_prediction = 70
# weight = 0.5  # начальное значение
# learning_rate = 0.001 #alpha-коэффициент
# print("Обучение нейросети:")
# for i in range(13):
#     prediction = neural_networks(inp, weight)
#     error = get_error(true_prediction, prediction)
#     print(f"Prediction: {prediction:.10f}, Weight: {weight:.5f}, Error: {error:.20f}")
    
#     delta = (prediction - true_prediction) * inp * learning_rate
#     weight = weight - delta

# # Финальный результат
# print(f"\nФинальная ошибка: {get_error(true_prediction, neural_networks(inp, weight)):.20f}")
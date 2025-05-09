import random
from collections import defaultdict
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model


# util function to convert a tensor into a valid image
def deprocess_image(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x = np.squeeze(x)  # (1, 224, 224, 3) → (224, 224, 3)
    # x = np.clip(x, 0, 1)
    x = (x * 255).astype('uint8')  # Optional: scale back to [0,255] if needed
      # Ensure pixel values are in [0, 255]
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    # new_grads = np.zeros_like(gradients)
    # new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    # start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
    #                                                  start_point[1]:start_point[1] + rect_shape[1]]
    # return new_grads
    new_grads = np.zeros_like(gradients)
    x1, y1 = start_point
    x2, y2 = x1 + rect_shape[0], y1 + rect_shape[1]
    new_grads[:, x1:x2, y1:y2] = gradients[:, x1:x2, y1:y2]
    return new_grads


def constraint_light(gradients):
    # new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return np.full_like(gradients, grad_mean)


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True
# @tf.function
# def update_coverage(input_data, model, model_layer_dict, threshold=0):
#     # 필요한 레이어 이름 필터링
#     layer_names = [layer.name for layer in model.layers if
#                    'flatten' not in layer.name and 'input' not in layer.name]
    
#     # 각 레이어별로 출력 얻기
#     for layer_name in layer_names:
#         intermediate_layer_model = tf.keras.Model(inputs=model.input,
#                                                   outputs=model.get_layer(layer_name).output)
        
#         # 한 번에 한 레이어씩 예측하여 메모리 절약
#         intermediate_layer_output = intermediate_layer_model(input_data)

#         # GPU 계산을 위한 텐서플로우 연산으로 평균 계산
#         scaled = tf.reduce_mean(intermediate_layer_output, axis=-1)
#         condition = scaled > threshold
#         condition = tf.squeeze(condition, axis=0)
#         # 레이어별 활성화 상태 업데이트
#         for num_neuron in range(condition.shape[-1]):
#             if tf.reduce_any(condition[...,num_neuron]) and not model_layer_dict.get((layer_name, num_neuron), False):                
#                 model_layer_dict[(layer_name, num_neuron)] = True

def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - tf.reduce_min(intermediate_layer_output)) / (
        tf.reduce_max(intermediate_layer_output) - tf.reduce_min(intermediate_layer_output))
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

def get_before_softmax_output(model, input_data):
    """ 모델에서 before_softmax 레이어 출력을 가져옴 """
    # 모델의 중간 레이어 출력 설정 (before_softmax 레이어)
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer('before_softmax').output)
    return intermediate_layer_model(input_data)
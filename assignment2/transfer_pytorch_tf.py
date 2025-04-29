import onnx
from onnx_tf.backend import prepare
import torchvision.models as models
import torch
import torch.nn as nn


model1 = models.resnet50(weights=None)
model1.fc = nn.Linear(2048, 10)  # CIFAR-10은 10클래스

model2 = models.resnet50(weights=None)
model2.fc = nn.Linear(2048, 10)

model3 = models.resnet50(weights=None)
model3.fc = nn.Linear(2048, 10)

ckpt1 = torch.load('cifar10_model/lightning_logs/version_0/checkpoints/epoch=29-step=4710.ckpt')
ckpt2 = torch.load('cifar10_model/lightning_logs/version_1/checkpoints/epoch=29-step=4710.ckpt')
ckpt3 = torch.load('cifar10_model/lightning_logs/version_2/checkpoints/epoch=29-step=4710.ckpt')

model1.load_state_dict(ckpt1['state_dict'], strict=False)
model2.load_state_dict(ckpt2['state_dict'], strict=False)
model3.load_state_dict(ckpt3['state_dict'], strict=False)

model1.eval()
model2.eval()
model3.eval()

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    model1, 
    dummy_input, 
    "model1.onnx",
    input_names=['input'],      # input 이름 명시
    output_names=['output'],    # output 이름 명시
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=11  # 안정적인 ONNX 변환 위해 설정 (8 이상 추천)
)

torch.onnx.export(
    model2, 
    dummy_input, 
    "model2.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=11
)

torch.onnx.export(
    model3, 
    dummy_input, 
    "model3.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
onnx_model1 = onnx.load("model1.onnx")  
onnx_model2 = onnx.load("model2.onnx")  
onnx_model3 = onnx.load("model3.onnx")  

tf_model1 = prepare(onnx_model1)  
tf_model2 = prepare(onnx_model2)
tf_model3 = prepare(onnx_model3)

tf_model1.export_graph("saved_model1")  
tf_model2.export_graph("saved_model2")
tf_model3.export_graph("saved_model3")
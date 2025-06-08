from maraboupy import Marabou
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch

# 1. CIFAR-10 이미지 불러오기 (첫 번째 이미지 사용)
transform = transforms.Compose([
    transforms.ToTensor()  # [0, 1] 범위의 (C, H, W) Tensor로 변환
])

cifar10 = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
img_tensor, label = cifar10[0]  # img_tensor: torch.Size([3, 32, 32])

# numpy 배열로 변환
img_np = img_tensor.numpy()  # shape: (3, 32, 32)
print(img_np.shape)  # (3, 32, 32)
# 2. ONNX 모델 불러오기
filename = "SmallerCNN_cifar10.onnx"
network = Marabou.read_onnx(filename)

# 3. 입력/출력 변수 설정
inputVars = network.inputVars[0][0]    # shape: (3, 32, 32)
outputVars = network.outputVars[0]     # shape: (10,)

# 4. Perturbation 설정
delta = 0.03
for c in range(3):
    for h in range(32):
        for w in range(32):
            val = img_np[c][h][w]
            var = inputVars[c][h][w]
            network.setLowerBound(var, max(0.0, val - delta))
            network.setUpperBound(var, min(1.0, val + delta))

# network.setLowerBound(outputVars[0][0], 6.0)

# 5. 출력 조건 설정 (예: 원래 레이블 유지되는지 확인)

target_class = label
outputVars = network.outputVars[0]  # shape: (1, 10)
if isinstance(outputVars[0], (list, tuple, np.ndarray)):
    outputVars = outputVars[0] 
for i in range(10):
    if i != target_class:
        network.addInequality([outputVars[target_class], outputVars[i]], [1, -1],  -0.05)

options = Marabou.createOptions(verbosity = 0)

# 6. 검증 실행
print(f"Running Marabou verification for CIFAR-10 image (label={target_class})...")
vals, stats,_  = network.solve()



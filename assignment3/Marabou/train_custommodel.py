import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn

class SmallerCNN(nn.Module):
    def __init__(self):
        super(SmallerCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),  # 3x32x32 → 4x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                           # → 4x16x16
            nn.Flatten(),
            nn.Linear(4 * 16 * 16, 10)
        )
    def forward(self, x):
        return self.net(x)
    
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),   # (3,32,32) → (8,32,32)
            nn.ReLU(),
            nn.AvgPool2d(2),                             # → (8,16,16)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # → (16,16,16)
            nn.ReLU(),
            nn.AvgPool2d(2)                              # → (16,8,8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 10)  # → 10 클래스
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



batch_size = 256
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 준비 (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] 정규화
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 3. 모델 정의
model = SmallerCNN().to(device)

# 4. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 학습 루프
print("Training SmallerCNN on CIFAR-10...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# 6. 모델 저장
# torch.save(model.state_dict(), "SmallerCNN_cifar10_2.pth")
# print("Saved model to SmallerCNN_cifar10_2.pth")

# 7. ONNX export
model.eval()
dummy_input = torch.randn(1, 3, 32, 32).to(device)
torch.onnx.export(model, dummy_input, "SmallerCNN_cifar10.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=13)
print("Exported model to SmallerCNN_cifar10.onnx")
import torch 
import torchvision
import torch.nn.functional as F 
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNIST_model,CIFAR_model
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

def fsgm_targeted(model,x,target,eps) : 
    x.requires_grad = True 
    output = model(x)
    loss  = F.cross_entropy(output,target)
    model.zero_grad()
    loss.backward() 
    sign_grad = x.grad.sign()
    x_adv = x - eps*sign_grad
    x_adv = torch.clamp(x_adv,0,1)
    return x_adv 

def fsgm_untargeted(model,x,label,eps) : 
    x.requires_grad = True 
    output = model(x)
    loss  = F.cross_entropy(output,label)
    model.zero_grad()
    loss.backward()
    sign_grad=x.grad.sign()
    x_adv = x + eps*sign_grad
    x_adv = torch.clamp(x_adv,0,1)
    return x_adv 

def pgd_targeted(model,x,target,k,eps,eps_step) : 
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    x_orig = x.clone().detach()

    for _ in range(k) : 
        output = model(x_adv) 
        loss = F.cross_entropy(output,target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            x_adv = x_adv-eps_step * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv,x_orig+eps),x_orig-eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv.requires_grad = True
    return x_adv 

def pgd_untargeted(model,x,target,k,eps,eps_step) : 
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    x_orig = x.clone().detach()

    for _ in range(k):
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv += eps_step * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)  # project
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv.requires_grad = True

    return x_adv

def evaluate_attack(model, test_loader, eps, eps_step, k, is_cifar=False, classes=None, save_path="result.png"):
    device = next(model.parameters()).device
    model.eval()
    
    # 공격 통계
    total = 0
    counters = {
        "fgsm_untargeted": 0,
        "pgd_untargeted": 0,
        "fgsm_targeted": 0,
        "pgd_targeted": 0,
    }

    last_sample = {}

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        if pred.item() != y.item():
            continue

        total += 1

        # Untargeted
        x_fgsm_u = fsgm_untargeted(model, x, y, eps)
        x_pgd_u = pgd_untargeted(model, x, y, k, eps, eps_step)

        with torch.no_grad():
            pred_fgsm_u = model(x_fgsm_u).argmax(dim=1)
            pred_pgd_u = model(x_pgd_u).argmax(dim=1)

        if pred_fgsm_u.item() != y.item():
            counters["fgsm_untargeted"] += 1
        if pred_pgd_u.item() != y.item():
            counters["pgd_untargeted"] += 1

        # Targeted
        target_class = 0 if y.item() != 0 else 1
        target = torch.tensor([target_class], device=device)

        x_fgsm_t = fsgm_targeted(model, x, target, eps)
        x_pgd_t = pgd_targeted(model, x, target, k, eps, eps_step)

        with torch.no_grad():
            pred_fgsm_t = model(x_fgsm_t).argmax(dim=1)
            pred_pgd_t = model(x_pgd_t).argmax(dim=1)

        if pred_fgsm_t.item() == target_class:
            counters["fgsm_targeted"] += 1
        if pred_pgd_t.item() == target_class:
            counters["pgd_targeted"] += 1

        last_sample = {
            "x": x, "y": y, "target_class": target_class,
            "x_fgsm_u": x_fgsm_u, "pred_fgsm_u": pred_fgsm_u,
            "x_fgsm_t": x_fgsm_t, "pred_fgsm_t": pred_fgsm_t,
            "x_pgd_u": x_pgd_u, "pred_pgd_u": pred_pgd_u,
            "x_pgd_t": x_pgd_t, "pred_pgd_t": pred_pgd_t,
        }

    print(f"Total evaluated samples: {total}")
    for k, v in counters.items():
        print(f"{k.replace('_', ' ').upper()} Success Rate: {v / total * 100:.2f}%")

    if total > 0:
        fig, axs = plt.subplots(1, 5, figsize=(18, 5))
        img = last_sample["x"].squeeze().cpu().detach()
        if is_cifar:
            img = img.permute(1, 2, 0)

        axs[0].imshow(img, cmap=None if is_cifar else "gray")
        axs[0].set_title(f"Original: {classes[last_sample['y'].item()]}" if is_cifar else f"Original: {last_sample['y'].item()}")

        axs[1].imshow(last_sample["x_fgsm_u"].squeeze().cpu().detach().permute(1, 2, 0) if is_cifar else last_sample["x_fgsm_u"].squeeze().cpu().detach(), cmap=None if is_cifar else "gray")
        axs[1].set_title(f"FGSM Untargeted: {classes[last_sample['pred_fgsm_u'].item()]}" if is_cifar else f"FGSM Untargeted: {last_sample['pred_fgsm_u'].item()}")

        axs[2].imshow(last_sample["x_fgsm_t"].squeeze().cpu().detach().permute(1, 2, 0) if is_cifar else last_sample["x_fgsm_t"].squeeze().cpu().detach(), cmap=None if is_cifar else "gray")
        axs[2].set_title(f"FGSM Targeted: {classes[last_sample['target_class']]} / Prediction : {classes[last_sample['pred_fgsm_t'].item()]}" if is_cifar else f"FGSM Targeted: {last_sample['target_class']} / Prediction : {last_sample['pred_fgsm_t'].item()}")

        axs[3].imshow(last_sample["x_pgd_u"].squeeze().cpu().detach().permute(1, 2, 0) if is_cifar else last_sample["x_pgd_u"].squeeze().cpu().detach(), cmap=None if is_cifar else "gray")
        axs[3].set_title(f"PGD Untargeted: {classes[last_sample['pred_pgd_u'].item()]}" if is_cifar else f"PGD Untargeted : {last_sample['pred_pgd_u'].item()}")

        axs[4].imshow(last_sample["x_pgd_t"].squeeze().cpu().detach().permute(1, 2, 0) if is_cifar else last_sample["x_pgd_t"].squeeze().cpu().detach(), cmap=None if is_cifar else "gray")
        axs[4].set_title(f"PGD Targeted: {classes[last_sample['target_class']]} / Prediction : {classes[last_sample['pred_pgd_t'].item()]}" if is_cifar else f"PGD Targeted: {last_sample['target_class']} / Prediction : {last_sample['pred_pgd_t'].item()}")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# Hyperparameters
eps = 0.25
eps_step = 0.03
k = 15

# MNIST 평가
mnist_model = MNIST_model.load_from_checkpoint('MNIST_model/lightning_logs/version_0/checkpoints/epoch=29-step=7050.ckpt')
transform = transforms.ToTensor()
MNIST_test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
MNIST_test_loader = DataLoader(MNIST_test_dataset, batch_size=1, shuffle=True)
MNIST_classes = [
    0,1,2,3,4,5,6,7,8,9
]     
evaluate_attack(mnist_model, MNIST_test_loader,eps,eps_step,k,save_path="MNIST_adv_result.png")

# CIFAR 평가
cifar_model = CIFAR_model.load_from_checkpoint('cifar10_model/lightning_logs/version_0/checkpoints/epoch=29-step=4710.ckpt')
cifar10_normalization = torchvision.transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)
cifar10_test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization,
    ]
)
cifar_test_dataset = dataset_test = CIFAR10("./data", train=False, download=True, transform=cifar10_test_transforms)
cifar_test_loader = DataLoader(cifar_test_dataset, batch_size=1, shuffle=True)
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
evaluate_attack(cifar_model, cifar_test_loader, eps, eps_step, k, is_cifar=True, classes=cifar10_classes, save_path="CIFAR_adv_result.png")

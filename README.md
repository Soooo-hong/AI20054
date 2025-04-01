# AI20054

## Installation 
I tested on cuda 11.8, gcc 11.2.
```
git clone https://github.com/Soooo-hong/AI20054.git
cd AI20054
conda create -n ai python=3.8 -y
conda activate ai

# Install the pytorch version for your cuda version.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install Other dependencies
pip install -r requirements.txt
```

## 1. Training
### Train
```
python train.py
```
If you run the command, you would download MNIST, CIFAR-10 datasets and train each model for adversirial attack. 
Two model classes in this project are implemented with reference to the source code and design patterns of PyTorch Lightning. 


## 2. Test
```
python test.py
```
Evaluating the success rate of the entire adversarial attack may take several minutes.
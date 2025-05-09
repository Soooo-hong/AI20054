# Assignment2

## Installation 
I tested on cuda 11.8, gcc 11.2.
```
git clone https://github.com/Soooo-hong/AI20054.git
cd AI20054/assignment2
conda create -n train python=3.7 -y
conda activate deepxplore
pip install -r requirements.txt
```

## 1. Training models
### Training and dataset donwloading
```
cd Cifar
conda activate train

python Model1_new_1_1.py && python Model2_new_2_2.py && python Model3__new_3_3.py
```
If you run the command, you would get three model weights of CIFAR-10 datasets. 
* ResNet50_CIFAR10_Model1.h5
* ResNet50_CIFAR10_Model2.h5
* ResNet50_CIFAR10_Model3.h5

## 2. Run DeepXplore on MNIST dataset
```
python gen_diff.py 
```
Then the generated_inputs are stored.
* assignment2/Cifar/generated_inputs

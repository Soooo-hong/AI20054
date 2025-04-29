# AI20054

## Installation 
I tested on cuda 11.8, gcc 11.2.
```
git clone https://github.com/Soooo-hong/AI20054.git
cd AI20054/assignment2
conda create -n ai python=3.8 -y
conda activate ai

# Install the pytorch version for your cuda version.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install Other dependencies
pip install -r requirements.txt
```

## 1. Training models
### Training and dataset donwloading
```
python train.py
```
If you run the command, you would download CIFAR-10 datasets and train each model for adversirial attack. 
```
    .
    └── datasets                    
        ├── cifar-10-batches-py 
            ├──  
        
```
The model classes in this project are implemented with reference to the source code and design patterns of PyTorch Lightning. 


## 2. Transfering pytorch to tensorflow
After traininig, each of the model's checkpoint is stored.
* cifar10_model/lightning_logs/version_0/checkpoints
* cifar10_model/lightning_logs/version_1/checkpoints
* cifar10_model/lightning_logs/version_2/checkpoints
```
python transfer_pytorch_tf.py
```
Then you can get saved model file 
* assignment2/saved_model1
* assignment2/saved_model2
* assignment2/saved_model3
```

## 3. Run DeepXplore on MNIST dataset
```
cd MNIST
python gen_diff.py
```
Then the generated_inputs are stored.
 ```
* assignment2/MNIST/generated_inputs
```

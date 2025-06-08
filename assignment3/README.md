# Assignment3

## Installation 
I tested on cuda 11.8, gcc 11.2.
```
git clone https://github.com/Soooo-hong/AI20054.git
cd AI20054/assignment3
conda create -n assignment3 python=3.8 -y
conda activate assignment3
pip install -r requirements.txt
```

## 2. Run Marabou for PointNet
```
cd Marabou
python train_custommodel.py
python test_pointnet.py 
```
To get onnx file, you should command python train_custommodel.py. Then you can obtain SmallerCNN_cifar10.onnx.
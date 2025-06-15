# Assignment4

## Installation 
I tested on cuda 12.1, gcc 11.2.
```
git clone https://github.com/Soooo-hong/AI20054.git
cd AI20054/assignment4/alpha-beta-crown 
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown
pip install -r complete_verifier/requirements.txt
```
I followed the instruction of alpha-beta crown installation

## 2. Run Alpha-Beta Crown for CustomNet
```
cd alpha-beta-crown/complete_verifier
python abcrown.py --config exp_configs/tutorial_examples/custom_cifar_simple.yaml
```

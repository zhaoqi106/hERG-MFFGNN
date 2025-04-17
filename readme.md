#### How to train your own model?
1. Pre-process your dataset by **molnetdata.py**.
2. Set your dataset name correctly at **main.py**, which ia suppose to be the same with the argument **--moldata** in **molnetdata.py**.
3. Run **main.py**
#### Environment
1. rdkit 2023.9.6
2. numpy 1.26.0
3. torch 2.1.2+cu121
4. torch_geometric 2.5.3
####Data 
The generated pt files are located in the hERG-MFFGNN/dataset/processed
The benchmark dataset and four external validation datasets are in the hERG-MFFGNN/dataset/raw
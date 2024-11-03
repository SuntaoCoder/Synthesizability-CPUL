# Synthesizability-CPLU

## Developers
Tao Sun<br>

## Prerequisites
python==3.9<br> 
pytorch==2.0.0<br>
pymatgen==2024.3.1<br>
pandas==2.2.1<br> 
numpy==1.26.4<br>
matplotlib==3.8.3<br>
seaborn==0.13.2<br>
scikit-learn==1.3.2<br>
tqmd==4.66.2<br>

## Usage
### [1] Define a customized dataset and generate crystal graphs
To input crystal structures to Synthesizability-CPUL, you will need to define a customized dataset and pre-generate crystal graph as pickle files. 
#### 1) id_prop.csv: a CSV file with two columns for positive data(synthesizable) and unlabeled data(not-yet-synthesized). The first column recodes a unique ID for each crystal, and the second column recodes the value (1 = positive, 0 = unlabeled) according to whether they were synthesized already or not.
#### 2) atom_init.json: a JSON file that stores the initialization vector for each element.
#### 3) id.cif: a CIF file that recodes the crystal structure, where ID is the unique ID for the crystal.

If you want to generate crystal graph with cutoff radius 6A, maximum 12 neighbors:<br>
`python generate_crystal_graph.py --cifs ./data/cifs --n 12 --r 6 --f ./saved_crystal_graph`<br>
Then, you will obtain preloaded crystal graph files in folder “saved_crystal_graph”<br>

### [2] Train a CGCL model
#### Train the CGCL model on a dataset consisting of 48,884 positive samples and 104,351 unlabeled samples.<br>
`python CGCL.py --id_prop_file 'id_prop.csv' --save_result 'log.txt' --save_model 'model.pt'
 `<br>

### [3] Extract features
#### Extract features for 48,884 positive samples and 104,351 unlabeled samples using CGCL.
`python get_embeddings.py --id_prop_file 'id_prop.csv' --load_model 'model.pt' --save_feature 'embedding.npy'
`<br>


### [4] Train a CPU-MLP model
#### Train a CPU-MLP model on a dataset consisting of 48,884 positive samples and 104,351 unlabeled samples.<br>
`python cpu_learning.py `<br>

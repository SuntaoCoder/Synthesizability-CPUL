from __future__ import print_function, division
import sys
import argparse
import csv
import functools
import os
import random
import warnings
from tqdm import tqdm
import json
import pickle

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

'''
Generation crystal graph of all data (positive + unlabeled) for PU-learning
'''


def one_hot(number, length):
    a = np.zeros(length)
    a[number-1] = 1
    a_list = a.tolist()
    return a_list


def one_hot_continuous(x, lower, upper, length):
    h = (upper - lower) / length
    for l in range(1, length + 1):
        if x <= lower + l * h:
            vec_list = one_hot(l, length)
            break
    return vec_list


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=0.5):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def preprocess(inputs):
    cif_id, target = inputs
    if os.path.exists(pickle_folder + '/' + cif_id + '.pickle'):
        return
    crystal = Structure.from_file(os.path.join(root_dir, cif_id + '.cif'))

    atom_attribute_fea = np.vstack([ari.get_atom_fea(crystal[j].specie.number)
                                    for j in range(len(crystal))])
    atom_site_fea = {}
    for i in range(len(crystal)):
        site_x = one_hot_continuous(crystal.frac_coords[i][0], 0.0, 1.0, 20)
        site_y = one_hot_continuous(crystal.frac_coords[i][1], 0.0, 1.0, 20)
        site_z = one_hot_continuous(crystal.frac_coords[i][2], 0.0, 1.0, 20)
        atom_site_fea[i] = np.array(site_x+site_y+site_z)
    atom_site_fea = np.vstack([atom_site_fea[k]
                              for k in range(len(crystal))])

    atom_fea = np.concatenate((atom_attribute_fea, atom_site_fea), axis=1)
    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True, numerical_tol=0.25)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

    nbr_fea = gdf.expand(nbr_fea)
    nbr_fea = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    target = torch.Tensor([float(target)])

    preload_data = ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
    with open(pickle_folder + '/' + cif_id + '.pickle', "wb") as f:
        pickle.dump(preload_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neighbor Information Preloader of Crystal Graph Convolutional '
                                                 'Neural Networks')
    parser.add_argument('--cifs', type=str, help='Root directory where .cif files exist')
    parser.add_argument('--n', type=int, default=12, help='The maximum number of neighbors while constructing the '
                                                          'crystal graph')
    parser.add_argument('--r', type=float, default=6, help='The cutoff radius for searching neighbors(A)')
    parser.add_argument('--dmin', type=float, default=0, help='The minimum distance for constructing GaussianDistance')
    parser.add_argument('--s', type=float, default=0.2, help='The step size for constructing GaussianDistance')
    parser.add_argument('--f', type=str, help='Folder name for saving crystal graph (.pickle files)')
    parser.add_argument('--idprop', type=str, default='id_prop.csv', help='id-prop.csv file')
    args = parser.parse_args()

    root_dir = args.cifs
    max_num_nbr = args.n
    radius = args.r
    fea_dict = {}

    assert os.path.exists(root_dir), 'root_dir does not exist!'
    id_prop_file = os.path.join(root_dir, args.idprop)
    assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    # random.seed(1234)
    # random.shuffle(id_prop_data)
    atom_init_file = os.path.join(root_dir, 'atom_init.json')
    assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=args.dmin, dmax=radius, step=args.s)

    pickle_folder = os.path.join(os.getcwd(), args.f)
    if not os.path.exists(pickle_folder):
        os.system('mkdir ' + pickle_folder)

    for i in tqdm(id_prop_data):
        preprocess(i)

    with open(pickle_folder + '/' + id_prop_data[0][0] + '.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data[0][0].shape)
        print(data[0][1].shape)
        print(data[0][2].shape)
        print(data[1])
        print(data[2])

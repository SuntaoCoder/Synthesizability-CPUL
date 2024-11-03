import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import pickle
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.cgcnn import collate_pool
import torch.nn as nn
from model.cgcnn import CrystalGraphConvNet
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR


def arg_parse():
    parser = argparse.ArgumentParser(description='Get embeddings by CGCL.')
    parser.add_argument('--device', default='cuda', type=str, help='gpu device ids')
    parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--alpha', default=1.2, type=float, help='stregnth for regularization')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')
    parser.add_argument('--seed', type=int, default=123456)
    # Random
    parser.add_argument('--eta', type=float, default=1.0, help='0.1, 1.0, 10')
    parser.add_argument('--batch_size', type=int, default=128, help='128, 256, 512, 1024')
    # CGCNN
    parser.add_argument('--graph', type=str, default="./saved_crystal_graph", metavar='N',
                        help='Folder name for preloaded crystal graph files')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                        help='number of hidden atom features in conv layers')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                        help='number of conv layers')
    parser.add_argument('--id_prop_file', default='id_prop2.csv', type=str, help='id_prop1.csv, id_prop2.csv, id_prop.csv, id_prop_test.csv')
    parser.add_argument('--load_model', default='model2.pt', type=str, help='model1.pt, model2.pt, model.pt')
    parser.add_argument('--save_feature', default='embedding1.npy', type=str, help='embedding1.npy, embedding2.npy, embedding.npy, embedding1_24.npy')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def preload(preload_folder, id_prop_file):
    data = []
    with open(id_prop_file) as g:
        reader = csv.reader(g)
        cif_list = [row[0] for row in reader]

    for cif_id in tqdm(cif_list):
        with open(preload_folder + '/' + cif_id + '.pickle', 'rb') as f:
            data.append(pickle.load(f))

    return data


class simclr(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len):
        super(simclr, self).__init__()
        self.embedding_dim = args.atom_fea_len
        self.encoder = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                           atom_fea_len=args.atom_fea_len,
                                           n_conv=args.n_conv)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        y = self.encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        y = self.proj_head(y)
        return y


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    device = torch.device(args.device)
    batch_size = args.batch_size
    graph_dir = args.graph
    collate_fn = collate_pool
    data_folder = './data'
    saved_model = './saved_model'
    feature_folder = './feature_analysis'
    
    dataset_test = preload(preload_folder=graph_dir,
                           id_prop_file=os.path.join(data_folder, args.id_prop_file))
    

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    structures, _, _ = dataset_test[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = simclr(orig_atom_fea_len, nbr_fea_len).to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(saved_model, args.load_model)))

    emb, _ = model.encoder.get_embeddings(test_loader, device)
    np.save(os.path.join(feature_folder, args.save_feature), emb)
    

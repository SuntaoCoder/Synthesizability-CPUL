import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import pickle
import csv
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from cgcnn.data_PU_learning import collate_pool
import torch.nn as nn
from model.cgcnn import CrystalGraphConvNet
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from utils import AverageMeter


def arg_parse():
    parser = argparse.ArgumentParser(description='CGCL')
    parser.add_argument('--device', default='cuda:0', type=str, help='gpu device ids')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--lr-milestones', default=[20], nargs='+', type=int,
                        metavar='N', help='milestones for scheduler (default: '
                                          '[100])')
    parser.add_argument('--alpha', default=1.2, type=float, help='stregnth for regularization')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--epochs', type=int, default=100)
    # Random
    parser.add_argument('--eta', type=float, default=1, help='0.1, 1.0, 10')
    parser.add_argument('--batch_size', type=int, default=256, help='128, 256, 512, 1024')
    # CGCNN
    parser.add_argument('--graph', type=str, default="./saved_crystal_graph", metavar='N',
                        help='Folder name for preloaded crystal graph files')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                        help='number of hidden atom features in conv layers')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                        help='number of conv layers')
    parser.add_argument('--id_prop_file', default='id_prop1.csv', type=str, help='id_prop1.csv, id_prop2.csv, id_prop.csv')
    parser.add_argument('--save_result', default='log1.txt', type=str, help='log1.txt, log2.txt, log.txt')
    parser.add_argument('--save_model', default='model1.pt', type=str, help='model1.pt, model2.pt, model.pt')
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
#                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        y = self.encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug, labels):
        T = 0.1
        batch_size, _ = x.size()
        x = nn.functional.normalize(x, dim=1)
        x_aug = nn.functional.normalize(x_aug, dim=1)

        inner_pdt_00 = torch.einsum('nc,mc->nm', x, x) / T
        inner_pdt_01 = torch.einsum('nc,mc->nm', x, x_aug) / T

        inner_pdt_01_exp = torch.exp(inner_pdt_01)
        inner_pdt_00_exp = torch.exp(inner_pdt_00)
        nll_mtx = inner_pdt_00_exp

        mask_label = labels

        eq_mask = torch.eq(mask_label, torch.t(mask_label))
        eq_mask_original = eq_mask.clone().detach()
        
        neg_eq_mask = ~eq_mask
        frac_up_mask = eq_mask.fill_diagonal_(0)
        frac_down_mask = neg_eq_mask.fill_diagonal_(0)

        nll_mtx1 = nll_mtx * frac_up_mask
        nll_mtx2 = nll_mtx * frac_down_mask

        # loss1
        pos_sim1 = inner_pdt_01_exp[range(batch_size), range(batch_size)]
        loss1 = pos_sim1 / (nll_mtx2.sum(dim=1))
        loss1 = - torch.log(loss1).mean()

        # loss2
        nll_mtx3 = nll_mtx1 / torch.sum(nll_mtx2, dim=1, keepdim=True)
        nll_mtx3[nll_mtx3 != 0] = - torch.log(nll_mtx3[nll_mtx3 != 0])
        similarity_scores = nll_mtx3

        a = eq_mask_original.sum(dim=1).tolist()
        if 1 in a:
            loss2 = torch.zeros(batch_size)
        else:
            loss2 = similarity_scores.sum(dim=1) / (eq_mask_original.sum(dim=1) - 1)
        loss2 = torch.mean(loss2)
        loss = loss1 + loss2
        return loss, loss1, loss2


def gen_ran_output(input_var, model, vice_model, args):
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(
                device)
    z2 = vice_model(*input_var)
    return z2


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    device = torch.device(args.device)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    graph_dir = args.graph
    collate_fn = collate_pool
    data_folder = './data'
    saved_model = './saved_model'

    if device == "cpu":
        pin_memory = False
    else:
        pin_memory = True

    starttime = time.time()
    dataset_train = preload(preload_folder=graph_dir,
                            id_prop_file=os.path.join(data_folder, args.id_prop_file))

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=pin_memory)

    structures, _, _ = dataset_train[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = simclr(orig_atom_fea_len, nbr_fea_len).to(device)
    vice_model = simclr(orig_atom_fea_len, nbr_fea_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    align = []
    uniform = []
    
    for epoch in range(1, epochs + 1):
        loss_all = AverageMeter()
        loss_all1 = AverageMeter()
        loss_all2 = AverageMeter()

        model.train()
        for i, (input, target, _) in enumerate(train_loader):
            if device == 'cpu':
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
                target = target.cpu()
            else:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
                target = target.cuda()

            x2 = gen_ran_output(input_var, model, vice_model, args)
            x1 = model(*input_var)
            loss_old, loss1, loss2 = model.loss_cal(x1, x2, target)

            optimizer.zero_grad()
            loss_old.backward()
            optimizer.step()

            loss_all.update(loss_old.data.cpu().item(), len(input[3]))
            loss_all1.update(loss1.data.cpu().item(), len(input[3]))
            loss_all2.update(loss2.data.cpu().item(), len(input[3]))
        with open(args.save_result, 'a') as f:
            print('Epoch {}, train_loss {}, train_loss1 {}, train_loss2 {}'.format(
                epoch, loss_all.avg, loss_all1.avg, loss_all2.avg), file=f, flush=True)
        scheduler.step()
        
    torch.save(model.state_dict(), os.path.join(saved_model, args.save_model))
    
    endtime = time.time()
    print('Time {:.3f} s'.format(endtime-starttime))

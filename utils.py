import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn import metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bootstrap_aggregating(bagging_size, folder, filename='_test-unlabeled_predict.csv', save=1, repeat='No'):

    predict_dict = {}

    print("Do bootstrap aggregating for %d models.............." % bagging_size)
    for i in range(1, bagging_size+1):
        df = pd.read_csv(os.path.join(folder, 'id_prop_bag_' + str(i) + filename), header=None)
        id_list = df.iloc[:, 0].tolist()
        pred_list = df.iloc[:, 2].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predict_dict:
                predict_dict[mat_id].append(float(pred_list[idx]))
            else:
                predict_dict[mat_id] = [float(pred_list[idx])]

    print("Writing CLscore file....")

    if repeat == 'No':
        save_filename = 'pu'+str(save)+'_test_results_ensemble_'+str(bagging_size)+'_models.csv'
    else:
        save_filename = 'pu'+str(save)+'_test_results_ensemble_'+str(bagging_size)+repeat+'_models.csv'

    with open(save_filename, "w") as g:
        g.write("id,CLscore,bagging")

        for key, values in predict_dict.items():
            if isinstance(key, int):
                key = str(key)
            g.write('\n')
            g.write(key + ',' + str(np.mean(np.array(values))) + ',' + str(len(values)))

    print("Done")


def preload_feature(preload_folder, id_prop_file, feature_dim=64):
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        label = [int(row[1]) for row in reader]

    with open(id_prop_file) as g:
        reader = csv.reader(g)
        embedding_index = [int(row[2]) for row in reader]

    embedding = np.load(preload_folder)
    data = np.zeros((len(embedding_index), feature_dim))
    for i, index in enumerate(embedding_index):
        data[i, :] = embedding[index, :]

    return data, label


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


# 定义神经网络分类模型
class MlpNet(nn.Module):
    def __init__(self, input_size, hidden_size, classification=True):
        super(MlpNet, self).__init__()
        self.classification = classification
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)

        if self.classification:
            self.fc_out = nn.Linear(hidden_size, 2)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)
        if self.classification:
            self.softmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                # torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))    
        x = self.dropout(x)
        
        out = self.fc_out(x)
        
        if self.classification:
            out = self.softmax(out)

        return out


class MlpNet_Fe(nn.Module):
    def __init__(self, input_size, hidden_size, classification=True):
        super(MlpNet_Fe, self).__init__()
        self.classification = classification
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        if self.classification:
            self.fc_out = nn.Linear(hidden_size, 2)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)
        if self.classification:
            self.softmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                # torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        
        out = self.fc_out(x)
        
        if self.classification:
            out = self.softmax(out)

        return out


def class_eval(prediction, target, test):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        if not test:
            try:
                auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
            except ValueError:
                auc_score = 0
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    if test:
        return accuracy, precision, recall, fscore
    else:
        return accuracy, precision, recall, fscore, auc_score

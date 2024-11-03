import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from utils import AverageMeter, bootstrap_aggregating, preload_feature, MlpNet, MyDataset, class_eval


def split_bagging(id_prop, start, bagging_size, folder):
    """
    Return 3 CSV files, (id, synthesizability, index in id_prop)
    """
    df = pd.read_csv(id_prop, header=None)

    exp = []
    exp_index = []
    vir = []
    vir_index = []
    for i in range(len(df)):
        if df[1][i] == 1:
            exp.append(df[0][i])
            exp_index.append(i)
        elif df[1][i] == 0:
            vir.append(df[0][i])
            vir_index.append(i)
        else:
            raise Exception("ERROR: prop value must be 1 or 0")

    positive = pd.DataFrame()
    positive[0] = exp
    positive[1] = [1 for _ in range(len(exp))]
    positive[2] = exp_index

    unlabeled = pd.DataFrame()
    unlabeled[0] = vir
    unlabeled[1] = [0 for _ in range(len(vir))]
    unlabeled[2] = vir_index

    os.makedirs(folder, exist_ok=True)

    # Sample negative data for training
    for i in tqdm(range(start, start + bagging_size)):
        # Sample positive data for validation and training
        valid_positive = positive.sample(frac=0.2)
        train_positive = positive.drop(valid_positive.index)
    
        # Randomly labeling to negative
        negative = unlabeled.sample(n=len(positive[0]))
        valid_negative = negative.sample(frac=0.2)
        train_negative = negative.drop(valid_negative.index)

        valid = pd.concat([valid_positive, valid_negative])
        valid.to_csv(os.path.join(folder, 'id_prop_bag_' + str(i + 1) + '_valid.csv'), mode='w', index=False,
                     header=False)

        train = pd.concat([train_positive, train_negative])
        train.to_csv(os.path.join(folder, 'id_prop_bag_' + str(i + 1) + '_train.csv'), mode='w', index=False,
                     header=False)

        # Generate unlabeled data
        test_unlabel = unlabeled.drop(negative.index)
        test_unlabel.to_csv(os.path.join(folder, 'id_prop_bag_' + str(i + 1) + '_test-unlabeled.csv'), mode='w',
                            index=False, header=False)


if __name__ == '__main__':
    id_prop_c = "./data/id_prop.csv"
    folder_c = "./split_cpul"
    bagging = 100
    preload_folder_c = "./feature_analysis/embedding.npy"
    batch_size = 256
    lr = 0.001
    epochs = 100
    save_result = 'cpu_log.txt'
    device = "cuda"

    split_bagging(id_prop_c, 0, bagging, folder_c)

    for bag in range(0, bagging):

        with open(save_result, 'a') as f:
            print('bagging {}'.format(bag + 1), file=f, flush=True)

        train_features, train_labels = preload_feature(preload_folder=preload_folder_c,
                                                       id_prop_file=os.path.join(folder_c, 'id_prop_bag_' + str(
                                                           bag + 1) + '_train.csv'))
        val_features, val_labels = preload_feature(preload_folder=preload_folder_c,
                                                   id_prop_file=os.path.join(folder_c, 'id_prop_bag_' + str(
                                                       bag + 1) + '_valid.csv'))
        unlabeled_features, unlabeled_labels = preload_feature(preload_folder=preload_folder_c,
                                                               id_prop_file=os.path.join(folder_c, 'id_prop_bag_' +
                                                                                         str(bag + 1) + '_test-unlabeled.csv'))

        train_dataset = MyDataset(train_features, train_labels)
        val_dataset = MyDataset(val_features, val_labels)
        unlabeled_dataset = MyDataset(unlabeled_features, unlabeled_labels)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

        model = MlpNet(input_size=64, hidden_size=128).to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        for epoch in range(1, epochs + 1):
            loss_all = AverageMeter()
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all.update(loss.data.cpu().item(), len(targets))
            with open(save_result, 'a') as f:
                print('Epoch {}, train_loss {}'.format(epoch, loss_all.avg), file=f, flush=True)
            scheduler.step()

            if epoch % 10 == 0:

                losses = AverageMeter()
                accuracies = AverageMeter()
                precisions = AverageMeter()
                recalls = AverageMeter()
                fscores = AverageMeter()
                auc_scores = AverageMeter()

                model.eval()
                for val_inputs, val_targets in val_dataloader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss_val = criterion(val_outputs, val_targets)
                    losses.update(loss_val.data.cpu().item(), len(val_targets))

                    accuracy, precision, recall, fscore, auc_score = \
                        class_eval(val_outputs.data.cpu(), val_targets.cpu(), test=False)

                    accuracies.update(accuracy, val_targets.size(0))
                    precisions.update(precision, val_targets.size(0))
                    recalls.update(recall, val_targets.size(0))
                    fscores.update(fscore, val_targets.size(0))
                    auc_scores.update(auc_score, val_targets.size(0))

                with open(save_result, 'a') as f:
                    print('Epoch {}\t val_loss {}\t accuracy {}\t precision {}\t recall {}\t F1 {}\t AUC {}\t'.format(
                        epoch, losses.avg, accuracies.avg, precisions.avg, recalls.avg,
                        fscores.avg, auc_scores.avg), file=f, flush=True)

                if epoch % 100 == 0:
                    with open("cpu_log_metrics.txt", 'a') as f:
                        print('Epoch {}\t val_loss {}\t accuracy {}\t precision {}\t recall {}\t F1 {}\t AUC {}\t'.format(
                            epoch, losses.avg, accuracies.avg, precisions.avg, recalls.avg,
                            fscores.avg, auc_scores.avg), file=f, flush=True)


        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()

        model.eval()
        predict_label = []
        test_preds = []
        for unlabeled_inputs, _ in unlabeled_dataloader:
            unlabeled_inputs = unlabeled_inputs.to(device)
            unlabeled_outputs = model(unlabeled_inputs)
            unlabeled_outputs = unlabeled_outputs.cpu()
            prediction = np.exp(unlabeled_outputs.detach().numpy())
            pred_label = np.argmax(prediction, axis=1)
            predict_label.extend(pred_label)

            assert prediction.shape[1] == 2
            test_preds += prediction[:, 1].tolist()

        df_unlabeled = pd.read_csv(os.path.join(folder_c, 'id_prop_bag_' + str(bag + 1) + '_test-unlabeled.csv'),
                                   header=None)
        df_unlabeled_new = pd.DataFrame()
        df_unlabeled_new[0] = df_unlabeled.iloc[:, 0]
        df_unlabeled_new[1] = predict_label
        df_unlabeled_new[2] = test_preds
        df_unlabeled_new.to_csv(os.path.join(folder_c, 'id_prop_bag_' + str(bag + 1) + '_test-unlabeled_predict.csv'),
                                mode='w', index=False, header=False)

    bootstrap_aggregating(bagging, folder_c, save='cpul')

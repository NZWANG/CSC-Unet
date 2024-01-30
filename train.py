# -*- encoding: utf-8 -*-
# Author  : Haitong

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import LoadDataset
from evalution_segmentaion import eval_semantic_segmentation
import Models import CSC_Unet
import cfg

if __name__ == "__main__":

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
    val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    net = CSC_Unet.CSC_UNet(cfg.DATASET[1], cfg.unfolding).to(device)


    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

    val_best_miou = [0]

    for epoch in range(cfg.EPOCH_NUMBER):

        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        net.train()
        train_loss = 0
        train_miou = 0

        for i, sample in enumerate(train_data):
            img_data = sample['img'].to(device)
            img_label = sample['label'].to(device)
            # forward
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]
            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]
            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_miou += eval_metrix['miou']

        print('|Train Loss|: {:.5f}\t|Train MIoU|: {:.5f}\t'.format(train_loss / len(train_data),
                                                                    train_miou / len(train_data)))

        # valid
        net.eval()
        eval_loss = 0
        eval_miou = 0

        with t.no_grad():

            for j, sample in enumerate(val_data):
                valImg = sample['img'].to(device)
                valLabel = sample['label'].long().to(device)
                out = net(valImg)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, valLabel)
                eval_loss += loss.item()
                pre_label = out.max(dim=1)[1].data.cpu().numpy()
                pre_label = [i for i in pre_label]
                true_label = valLabel.data.cpu().numpy()
                true_label = [i for i in true_label]
                eval_metrics = eval_semantic_segmentation(pre_label, true_label)
                eval_miou += eval_metrics['miou']

            if max(val_best_miou) <= eval_miou / len(val_data):
                val_best_miou.append(eval_miou / len(val_data))
                t.save(net.state_dict(), "CSC_Unet.pth")

            print('|Valid Loss|: {:.5f}\t|Valid MIoU|: {:.5f}'.format(eval_loss / len(val_data),
                                                                      eval_miou / len(val_data)))

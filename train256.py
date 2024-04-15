import argparse
from itertools import cycle

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader.DRD_dataset import DRD_dataset
from loss.ProSupCon import ProSupConLoss
from loss.SupContrast import SupConLoss

from models.ConClsNet import ConCsNet
from utils.lr_scheduler import LR_Scheduler
from utils.utils import AverageMeter
from utils.metrics import metrics_cal
import os


def data_info(list_path):
    num = np.zeros(5)
    # list_path = 'data/train1.list'
    img_list = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            img_list.append(line)
    weights = np.zeros(len(img_list))
    for i in img_list:
        t = int(i.split('_')[-1])
        num[t] += 1
        # print(t)
    index_w = sum(num) / num
    # print(index_w)
    for id, i in enumerate(img_list):
        t = int(i.split('_')[-1])
        weights[id] = index_w[t]

    return weights


def main(args):
    experiment_name = ''
    model_save_path = 'saves/dwad.pt'


    train_list_path = 'data/train1.list'
    weights_cls = data_info(train_list_path)
    # weights_con = np.ones(len(weights_cls))

    sampler_cls = torch.utils.data.sampler.WeightedRandomSampler(weights_cls, len(weights_cls), replacement=True)
    # sampler_con = torch.utils.data.sampler.WeightedRandomSampler(weights_con, length, replacement=True)
    dataset_con = DRD_dataset('data/DRD256/train', train_list_path, 'contrast')
    trainloader_con = torch.utils.data.DataLoader(dataset_con,shuffle=True,  num_workers=args.workers,
                                                  batch_size=args.batch_size,
                                                  pin_memory=True)
    dataset_cls = DRD_dataset('data/DRD256/train', train_list_path, 'classification')
    trainloader_cls = torch.utils.data.DataLoader(dataset_cls, sampler=sampler_cls, num_workers=args.workers,
                                                  batch_size=args.batch_size,
                                                  pin_memory=True)

    dataset1 = DRD_dataset('data/DRD256/train', 'data/test.list', 'test')
    testloader = torch.utils.data.DataLoader(dataset1, num_workers=0, batch_size=16, pin_memory=True)

    model = ConCsNet(args.classes, 512, args.contrast_fea_size).to(args.device)
    # model.load_state_dict(di)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # criterion1 = SupConLoss().to(args.device)
    criterion1 = ProSupConLoss(args).cuda()
    criterion2 = torch.nn.CrossEntropyLoss()

    scheduler = LR_Scheduler('cos', args.lr, args.epochs, len(weights_cls), min_lr=1e-5, )
    writer = SummaryWriter('runs/' + experiment_name)
    best_acc = 0
    for epoch in range(args.epochs):
        # print('----------------------------------------------------------------------------------------------', epoch)

        loss, loss1, loss2 = train(model, trainloader_con, trainloader_cls, epoch, criterion1, criterion2, scheduler,
                                   optimizer, args)
        print('\n Epoch: {0}\t'
              'Training Loss {train_loss:.4f} \t'
              'Training Loss1 {train_loss1:.4f} \t'
              'Training Loss2 {train_loss2:.4f} \t'
              .format(epoch + 1, train_loss=loss, train_loss1=loss1, train_loss2=loss2))
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('training_loss1', loss1, epoch)
        writer.add_scalar('training_loss2', loss2, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        torch.save(model.state_dict(), model_save_path)

        if epoch % 2 == 0:
            # evaluate for one epoch
            acc, pr, re, f1 = valid(model, testloader, epoch, args)

            print('Epoch: {0}\t'
                  'Validation acc {acc:.4f} \t'
                  'Validation pr {pr:.4f} \t'
                  'Validation re {re:.4f} \t'
                  'Validation f1 {f1:.4f} \t'
                  .format(epoch, acc=acc, pr=pr, re=re, f1=f1))

            if best_acc < acc:
                acc = acc
                save_dict = {"net": model.state_dict()}
                torch.save(save_dict, os.path.join('saves', f"best.pth"))
            writer.add_scalar('acc', acc, epoch)
            writer.add_scalar('pr', pr, epoch)
            writer.add_scalar('re', re, epoch)
            writer.add_scalar('f1', f1, epoch)
            # save model
            save_dict = {"net": model.state_dict()}
            torch.save(save_dict, os.path.join('saves', "latest.pth"))


def train(model, trainloader_con, trainloader_cls, epoch, criterion1, criterion2, scheduler, optimizer, args):
    model.train()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    a = 1 - (epoch / args.epochs)
    for batch_idx, (data1, data2) in enumerate(zip(trainloader_con, trainloader_cls)):
        scheduler(optimizer, batch_idx, epoch)

        img1 = data1['img'].to(args.device)
        label1 = data1['label'].to(args.device)
        fea, _ = model(img1)
        loss1 = criterion1(fea, labels=label1)

        img2 = data2['img'].to(args.device)
        label2 = data2['label'].to(args.device)
        _, soft = model(img2)
        loss2 = criterion2(soft, label2)
        bsz = img2.shape[0]

        loss = loss1 * a + (1 - a) * loss2
        # loss1 = loss2
        # loss = loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)
        losses1.update(loss1.item(), bsz)
        losses2.update(loss2.item(), bsz)
        print(
            f"epoch:{epoch}, batch:{batch_idx}/{len(trainloader_con)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}, loss1:{losses1.avg:.4f}, loss2:{losses2.avg:.4f}")
    return losses.avg, losses1.avg, losses2.avg


def valid(model, testloader, epoch, args):
    model.eval()
    # accs = np.array([0, 0, 0, 0, 0])
    # sum1 = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    # acces = AverageMeter()
    outputs_train = []
    targets_train = []
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            img = data['img'].to(args.device)
            label = data['label'].to(args.device)
            _, soft = model(img)
            bsz = img.shape[0]
            out = torch.nn.functional.softmax(soft, dim=1)
            out = torch.argmax(out, dim=1)
            # print(out)
            # print(label)
            outputs_train.extend(item.cpu().detach().numpy() for item in out)
            targets_train.extend(item.cpu().detach().numpy() for item in label)

            # print(accs/sum1)
            # acces.update(acc, bsz)
            # print(f"Validation epoch:{epoch}, batch:{batch_idx}/{len(testloader)}, mean Dice:{acc}__{accs / sum1}")
        accuracy, precision, recall, f1, conf_matrix, measure_result = metrics_cal(outputs_train, targets_train)
        print(measure_result)
    return accuracy, np.mean(precision), np.mean(recall), np.mean(f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="data/ACDC/train/unlabel")
    parser.add_argument("--contrast_fea_size", type=int, default=512)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--save', metavar='SAVE', default='./saves', help='saved folder')
    parser.add_argument('--model_save_name', type=str, help='Saving path', default='con.pt')
    args = parser.parse_args()

    main(args)

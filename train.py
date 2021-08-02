# -*- coding:utf-8 _*-
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import NTIRE_Training_Dataset, NTIRE_Validation_Dataset
from graphs.loss.muloss import mu_loss
from graphs.adnet import ADNet
from utils.utils import *
from utils.metrics import normalized_psnr, psnr_tanh_norm_mu_tonemap


def get_args():
    parser = argparse.ArgumentParser(description='ADNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=int, default=0,
                        help="dataset: ntire dataset")
    parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory')
    parser.add_argument('--logdir', type=str, default='./checkpoints',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    # Training
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=2,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_decay_interval', type=int, default=100,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='training batch size (default: 4)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    end = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                             batch_data['input2'].to(device)
        label = batch_data['label'].to(device)
        pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                epoch,
                batch_idx * len(batch_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time
            ))


def validation(args, model, device, val_loader, optimizer, epoch, cur_psnr):
    model.eval()
    n_val = len(val_loader)
    val_psnr = 0
    val_mulaw = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            psnr_pred = torch.squeeze(pred.clone())
            psnr_label = torch.squeeze(label.clone())
            psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32).clip(0, 100)
            psnr_label = psnr_label.data.cpu().numpy().astype(np.float32)
            psnr = normalized_psnr(psnr_pred, psnr_label, psnr_label.max())
            mu_law = psnr_tanh_norm_mu_tonemap(psnr_pred, psnr_label)
            val_psnr += psnr
            val_mulaw += mu_law
    val_mulaw /= n_val
    val_psnr /= n_val
    print('Validation set: Average PSNR: {:.4f}, mu_law: {:.4f}'.format(val_psnr, val_mulaw))

    # capture metrics
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.logdir, 'val_latest_checkpoint.pth'))
    if val_mulaw > cur_psnr[0]:
        torch.save(save_dict, os.path.join(args.logdir, 'best_checkpoint.pth'))
        cur_psnr[0] = val_mulaw
        with open(os.path.join(args.logdir, 'best_checkpoint.json'), 'w') as f:
            f.write('best epoch:' + str(epoch) + '\n')
            f.write('Validation set: Average PSNR: {:.4f}, mu_law: {:.4f}\n'.format(val_psnr, val_mulaw))


def main():
    # settings
    args = get_args()
    # random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # model architectures
    model = ADNet(6, 5, 64, 32)
    cur_psnr = [0]
    # init
    if args.init_weights:
        init_parameters(model)
    # loss
    loss_dict = {0: nn.L1Loss, 1: nn.MSELoss, 2: mu_loss}
    criterion = loss_dict[args.loss_func]()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # dataset and dataloader
    train_dataset = NTIRE_Training_Dataset(root_dir=args.dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_dataset = NTIRE_Validation_Dataset(root_dir=args.dataset_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        validation(args, model, device, val_loader, optimizer, epoch, cur_psnr)


if __name__ == '__main__':
    main()

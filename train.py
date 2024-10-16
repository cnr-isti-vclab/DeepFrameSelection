#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model_video import *
from dataloader import *
from util import *

#
#training function
#
def train(model, device, loader, optimizer, epoch, batch_size):
    model.train()

    progress = tqdm(loader)
    total_loss = 0.0
    counter = 0

    for X,y in progress:
        X = X.to(device)
        y = y.to(device)

        if batch_size > 1:
            y = y.unsqueeze(1)

        optimizer.zero_grad()

        output = model(X)

        loss = F.mse_loss(output, y)
        total_loss += loss.item()
        counter += 1
       
        loss.backward()
        optimizer.step()

        progress.set_postfix({'loss': total_loss / counter})

    if counter > 0:
        total_loss /= counter

    return total_loss

#
#validation function
#
def validation(model, device, loader):
    model.eval()

    progress = tqdm(loader)
    total_loss = 0.0
    counter = 0

    targets = []
    predictions = []

    for X, y in progress:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            
            loss = F.mse_loss(output, y)
            
            targets.append(y)
            predictions.append(output)
            
            total_loss += loss.item()
            counter += 1

            progress.set_postfix({'loss': total_loss / counter})

    if counter > 0:
        total_loss /= counter

    targets = torch.cat(targets, 0).squeeze()
    predictions = torch.cat(predictions, 0).squeeze()

    return total_loss, targets, predictions
    
#
#
#main
#
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='Path to data dir')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('-diff', '--differential', type=int, default=0, help='Video Type (0 --> no differential encoding; 1 --> differential encoding')
    parser.add_argument('-mtd', '--method', type=str, default='our', help='Method to fit')
    parser.add_argument('--resume', default=None, help='Path to initial weights')
    args = parser.parse_args()
    
    res_size_x, res_size_y = getResolution()
    
    params = vars(args)
    params['dataset'] = os.path.basename(os.path.normpath(args.data))

    run_name = 'dfs_lr{0[lr]}_e{0[epochs]}_b{0[batch]}_d{0[differential]}_m_{0[method]}'.format(params, args.resume is not None)

    print("Creating dirs...")
    run_dir = os.path.join(args.runs, run_name)
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    print(run_dir)
    print(ckpt_dir)
    mkdir_s(run_dir)
    mkdir_s(ckpt_dir)

    batch_size = args.batch
    print("Batch size: " + str(batch_size))
    print("Differential: " + str(args.differential))
    print("Method: " + args.method)

    #CPU or CUDA?
    bCuda = torch.cuda.is_available()                   # do we have a CUDA GPU?
    device = torch.device("cuda" if bCuda else "cpu")   # use CPU or GPU

    # list all data files
    group = 7
    fps = 30
    fq_vec, img_vec = ReadDataset(args.data, group, args.method, fps)
    
    # train, test split
    train_data, val_data = split_data(fq_vec, group, fps)

    transform = getTransform(res_size_x, res_size_y, args.data, args.differential, True)

    train_data = DatasetModelVideo(train_data, img_vec, fps, transform, group, args.differential)
    val_data   = DatasetModelVideo(val_data,   img_vec, fps, transform, group, args.differential)

    train_loader = DataLoader(train_data, shuffle=True,  batch_size=args.batch, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_data,   shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    #init training
    best_mse = None
    ckpt_prev = ''
    t_c_loss = []
    t_v_loss = []

   
    start_epoch = 0
    if args.resume:
        if args.resume == 'same':
            folder = run_dir
        else:
            folder = args.resume
    
        model = ModelVideo(folder, device, args.differential)

        try:
            start_epoch = ckpt['epoch'] + 1
            best_mse = ckpt['mse']
        except:
            best_mse = None
            start_epoch = 0
        
        log_file = os.path.join(ckpt_dir, 'log_'+ str(start_epoch)+'.csv')
        log = pd.DataFrame()
    else:
        log_file = os.path.join(ckpt_dir, 'log.csv')
        log = pd.DataFrame()
        model = ModelVideo('', device, args.differential)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    #for each epoch
    for epoch in range(start_epoch, args.epochs):
        #train
        cur_loss = train(model, device, train_loader, optimizer, epoch, batch_size)
        #check validation
        val_loss, targets_v, predictions_v = validation(model, device, val_loader)

        cur_loss = float(cur_loss)
        val_loss = float(val_loss)
        metrics = {'epoch': epoch}
        metrics['cur_loss'] = cur_loss
        metrics['val_loss'] = val_loss
        log = log._append(metrics, ignore_index=True)
        log.to_csv(log_file, index=False)

        t_c_loss.append(cur_loss)
        t_v_loss.append(val_loss)
        plotGraph(t_c_loss, t_v_loss, ckpt_dir, True)
        plotGraph(t_c_loss, t_v_loss, './', True)

        if (best_mse is None) or (val_loss < best_mse) or (epoch == (args.epochs - 1)):
            best_mse = val_loss
            
            delta = (targets_v - predictions_v)
            errors = delta.cpu().numpy()
                        
            sz = errors.shape
            errors = np.reshape(errors, (sz[0], 1))
            
            #targets_v = targets_v.cpu().numpy()
            #predictions_v = predictions_v.cpu().numpy()
            #predictions_v = np.reshape(predictions_v, (sz[0], 1))
            #targets_v = np.reshape(targets_v, (sz[0], 1))
            #mtx = np.concatenate((targets_v, predictions_v, errors), axis=1)            
            #np.savetxt(os.path.join(run_dir, 'errors_' + args.method + '.txt'), mtx, fmt='%f')
            #np.savetxt(os.path.join('errors_' + args.method + '.txt'), mtx, fmt='%f')            
            
            plt.clf()
            sns.histplot(errors)
            plt.savefig('hist_errors_test_' +  args.method + '.png')
            plt.savefig(os.path.join(run_dir, 'hist_errors_test_' +  args.method + '.png'))
            
            
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                       'epoch': epoch,
                       'cur_loss': cur_loss,
                       'val_loss': val_loss,
                       'cnn_model': model.cnn_encoder.state_dict(),
                       'lstm_model': model.lstm_decoder.state_dict(),
                       'differential': args.differential,
                       }, ckpt)

            if (epoch == (args.epochs - 1)):
                sys.exit()

            if ckpt_prev:
                if os.path.isfile(ckpt_prev):
                    os.remove(ckpt_prev)
                       
            ckpt_prev = ckpt

        scheduler.step(cur_loss)

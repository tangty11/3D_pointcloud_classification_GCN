from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from data import ModelNet40
from model import GCN, GCNResNet, MultiHeadAttentionGCN, DenseResGCN, DilatedGCN
from baselines import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

def train_epoch(net, current_epoch, train_dataloader, cfg, dev, opt, writer, scaler):
    net.train()
    running_loss, running_correct = 0, 0
    num_batches = len(train_dataloader)
    
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {current_epoch}/{cfg.epochs}"
    )
    
    for batch_idx in progress_bar:
        batch_data = next(iter(train_dataloader))
        points, target = batch_data[0].to(dev), batch_data[1].to(dev).squeeze()
        if cfg.model_name in ['pointnet', 'dgcnn']:
            points = points.permute(0, 2, 1)
            
        opt.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            output = net(points)
            batch_loss = F.cross_entropy(output, target)
        
        # 使用 scaler 进行反向传播和优化
        scaler.scale(batch_loss).backward()
        scaler.step(opt)
        scaler.update()
        
        running_loss += batch_loss.item()
        running_correct += output.max(1)[1].eq(target).sum().item()
    
    epoch_loss = running_loss / num_batches
    epoch_acc = running_correct / len(train_dataloader.dataset)
    
    # Log to tensorboard
    writer.add_scalar('Train/Loss', epoch_loss, current_epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, current_epoch)
    
    return epoch_loss, epoch_acc

def validate_epoch(net, current_epoch, val_dataloader, cfg, dev, writer):
    net.eval()
    running_loss, running_correct = 0, 0
    num_batches = len(val_dataloader)

    progress_bar = tqdm(
        range(num_batches),
        desc=f"Validation Epoch {current_epoch}/{cfg.epochs}"
    )

    for batch_idx in progress_bar:
        batch_data = next(iter(val_dataloader))
        points, target = batch_data[0].to(dev), batch_data[1].to(dev).squeeze()
        if cfg.model_name in ['pointnet', 'dgcnn']:
            points = points.permute(0, 2, 1)

        with torch.no_grad(), autocast():
            output = net(points)
            batch_loss = F.cross_entropy(output, target)
        
        running_loss += batch_loss.item()
        running_correct += output.max(1)[1].eq(target).sum().item()
    
    epoch_loss = running_loss / num_batches
    epoch_acc = running_correct / len(val_dataloader.dataset)
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', epoch_loss, current_epoch)
    writer.add_scalar('Val/Accuracy', epoch_acc, current_epoch)
    
    return epoch_loss, epoch_acc

def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    
    # Basic training settings
    parser.add_argument('--exp_name', type=str, default='experiment_1', help='experiment name')
    parser.add_argument('--model_name', type=str, default='gcnresnet', 
                        choices=['gcn', 'gcnresnet', 'pointnet', 'dgcnn', 'dilatedgcn', 'multiheadgcn', 'densegcn'],
                        help='model architecture')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=16, help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='use SGD optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='number of points delivered for model training/testing')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='embedding dimensions')
    parser.add_argument('--k', type=int, default=20, 
                        help='k nearest neighbors for constructing adjacency matrix')
    parser.add_argument('--model_path', type=str, default=None, help='pretrained model path')
    
    # Model specific settings
    parser.add_argument('--use_smooth', action='store_true', default=False, help='use smooth')
    parser.add_argument('--gcn_layers', type=int, default=5, help='number of GCN layers')
    parser.add_argument('--address_overfitting', action='store_true', default=True, help='address overfitting')
    
    # GCNResNet specific settings
    parser.add_argument('--use_resnet', action='store_true', default=False, help='use ResNet architecture')
    parser.add_argument('--res_in', type=int, default=3, help='ResNet input dimension')
    parser.add_argument('--res_hid', type=int, default=64, help='ResNet hidden dimension')
    parser.add_argument('--res_out', type=int, default=40, help='ResNet output dimension')
    parser.add_argument('--res_num_blocks', type=int, default=6, help='number of ResNet blocks')

    # # DilatedGCN specific parameters
    # parser.add_argument('--initial_dilation', type=int, default=1, help='initial dilation rate')
    # parser.add_argument('--max_dilation', type=int, default=4, help='maximum dilation rate')

    # multihead gcn specific parameters
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout rate')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup tensorboard
    log_dir = os.path.join('runs', args.exp_name)
    writer = SummaryWriter(log_dir)
    
    # Setup data loaders with pin_memory=True
    train_loader = DataLoader(
        ModelNet40(partition='train', num_points=args.num_points),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points),
        num_workers=8,
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Initialize model and move to device
    model_dict = {
        'pointnet': PointNet,
        'dgcnn': DGCNN,
        'gcn': GCN,
        'gcnresnet': GCNResNet,
        'dilatedgcn': DilatedGCN,
        'multiheadgcn': MultiHeadAttentionGCN,
        'densegcn': DenseResGCN
    }
    
    if args.model_name not in model_dict:
        raise ValueError(f"Model {args.model_name} not implemented")
    
    model = model_dict[args.model_name](args).to(device)
    print(f"Using {torch.cuda.device_count()} GPUs!")
    
    # Setup optimizer
    if args.use_sgd:
        print("Using SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Using Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    if not args.eval:
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                net=model,
                current_epoch=epoch,
                train_dataloader=train_loader,
                cfg=args,
                dev=device,
                opt=optimizer,
                writer=writer,
                scaler=scaler
            )
            
            val_loss, val_acc = validate_epoch(
                net=model,
                current_epoch=epoch,
                val_dataloader=test_loader,
                cfg=args,
                dev=device,
                writer=writer
            )
            
            # Save checkpoint if validation accuracy improves
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(log_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                    'best_acc': best_acc
                }, checkpoint_path)
            
            scheduler.step()
            
    else:
        # Evaluation mode
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model path {args.model_path} does not exist")
            
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        class_correct = np.zeros(40)
        class_total = np.zeros(40)
        
        test_loader = DataLoader(
            ModelNet40(partition='test', num_points=args.num_points),
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device).squeeze()
                
                if args.model_name in ['pointnet', 'dgcnn']:
                    data = data.permute(0, 2, 1)
                
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                label_idx = label.item()
                class_total[label_idx] += 1
                if predicted.item() == label_idx:
                    class_correct[label_idx] += 1

        # Calculate and save per-class accuracy
        class_accuracy = class_correct / class_total
        for i in range(40):
            print(f'Class {i}: Accuracy {class_accuracy[i] * 100:.2f}%')
        
        pd.DataFrame(class_accuracy).to_csv(os.path.join(log_dir, 'class_accuracy.csv'))
    
    writer.close()

if __name__ == "__main__":
    main()
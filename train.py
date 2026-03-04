import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model import H3M, H3MLoss
from prediction_dataset import FootballPredictionDataset
import argparse
from tqdm import tqdm
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_kl_loss = 0
    
    for batch in tqdm(dataloader, desc='Training', disable=args.no_tqdm):
        obs = batch['obs'].to(device)
        fut = batch['fut'].to(device)
        
        optimizer.zero_grad()
        
        pred, kl_loss = model(obs, fut)
        
        loss, pred_loss, kl_loss = criterion(pred, fut, kl_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        if kl_loss is not None:
            total_kl_loss += kl_loss.item()
    
    n = len(dataloader)
    return total_loss/n, total_pred_loss/n, total_kl_loss/n


def eval_epoch(model, dataloader, device, dataset):
    model.eval()
    total_ade = 0
    total_fde = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', disable=args.no_tqdm):
            obs = batch['obs'].to(device)
            fut = batch['fut'].to(device)
            
            pred = model(obs)
            

            pred_denorm = dataset.denormalize(pred.cpu().numpy())
            fut_denorm = dataset.denormalize(fut.cpu().numpy())
            
            diff = pred_denorm - fut_denorm
            dist = np.linalg.norm(diff, axis=-1)
            
            ade = np.mean(dist)
            fde = np.mean(dist[:, -1])
            
            total_ade += ade * len(obs)
            total_fde += fde * len(obs)
            count += len(obs)
    
    return total_ade/count, total_fde/count


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = FootballPredictionDataset(
        args.data_path, 'train', args.obs_len, args.pred_len
    )
    test_dataset = FootballPredictionDataset(
        args.data_path, 'test', args.obs_len, args.pred_len
    )
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    model = H3M(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_agents=args.num_agents
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_step,
        gamma=args.decay_gamma
    )

    criterion = H3MLoss(lambda_kl=0.1)
    

    best_ade = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")
        
        train_loss, pred_loss, kl_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Train - Loss: {train_loss:.4f}, Pred: {pred_loss:.4f}, KL: {kl_loss:.4f}")
        
        test_ade, test_fde = eval_epoch(model, test_loader, device, test_dataset)
        print(f"Test - ADE: {test_ade:.3f}, FDE: {test_fde:.3f}")
        
        scheduler.step()

        save_dir = os.path.join(args.save_path, str(args.pred_len))
        os.makedirs(save_dir, exist_ok=True)
        if test_ade < best_ade:
            best_ade = test_ade
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_ade': best_ade
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model with ADE {best_ade:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--data_path', type=str, default='./base_datasets/football',
                        help='Path to data directory containing train_clean.p and test_clean.p')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=16)
    parser.add_argument('--num_agents', type=int, default=23)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--decay_step', type=int, default=20,
                        help='Step size for learning rate decay')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='Gamma (multiplicative factor) for learning rate decay')
    parser.add_argument('--no_tqdm', action='store_true',
                    help='Disable tqdm progress bar display')


    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_path, exist_ok=True)

    set_seed(42)

    main(args)
 
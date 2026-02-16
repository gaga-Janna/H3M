import torch
import numpy as np
import pickle
import os
from model import UniTrajPredictor
# from unitraj_full import UniTrajPredictor, UniTrajLoss
from prediction_dataset import FootballPredictionDataset
import argparse
from tqdm import tqdm

def compute_metrics(pred, gt, dataset):
    """计算详细的评估指标"""
    # 反归一化
    pred_denorm = dataset.denormalize(pred.cpu().numpy())
    gt_denorm = dataset.denormalize(gt.cpu().numpy())
    
    # 计算误差
    errors = np.linalg.norm(pred_denorm - gt_denorm, axis=-1)  # [T, N]
    # print(f"Pred shape: {pred_denorm.shape}")
    # print(f"GT shape: {gt_denorm.shape}")
    # print(f"Errors shape: {errors.shape}")
    # print(f"Sample errors: {errors[0, :5]}")  # 打印前5个智能体第一帧的误差
    
    # 整体指标
    ade = np.mean(errors)
    fde = np.mean(errors[-1])
    
    # 分组指标
    ball_ade = np.mean(errors[:, 0])
    ball_fde = errors[-1, 0]
    
    offense_ade = np.mean(errors[:, 1:12])
    offense_fde = np.mean(errors[-1, 1:12])
    
    defense_ade = np.mean(errors[:, 12:])
    defense_fde = np.mean(errors[-1, 12:])
    
    return {
        'ade': ade,
        'fde': fde,
        'ball_ade': ball_ade,
        'ball_fde': ball_fde,
        'offense_ade': offense_ade,
        'offense_fde': offense_fde,
        'defense_ade': defense_ade,
        'defense_fde': defense_fde,
        'errors': errors  
    }



def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载测试数据
    print("Loading test dataset...")
    test_dataset = FootballPredictionDataset(
        args.data_path,
        split='test',
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 加载模型
    print("Loading model...")

    model = UniTrajPredictor(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_agents=args.num_agents
    ).to(device)
    
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # 测试所有样本
    num_samples = min(args.num_samples, len(test_dataset))
    all_metrics = []
    all_predictions = []
    
    print(f"\nTesting {num_samples} samples...")
    




    # 统计器（Best-of-K，使用反归一化后的单位）
    bo_total_ade = 0.0
    bo_total_fde = 0.0
    bo_count = 0


    for i in tqdm(range(num_samples)):
        sample = test_dataset[i]
        obs = sample['obs'].unsqueeze(0).to(device)   # [1, T_obs, N, 2]
        fut = sample['fut'].unsqueeze(0).to(device)   # [1, T_pred, N, 2]

        with torch.no_grad():
            if args.use_best_k:
                pred = model.inference_best_of_k(
                    obs,
                    k=args.num_trajectories,
                    ground_truth=fut,
                    selection_mode='best',
                    temperature=1.0
                )  # [1, T, N, 2]
            else:
                pred = model(obs)

        # === 统一用反归一化后的误差来做 Best-of-K 的平均统计 ===
        pred_denorm = test_dataset.denormalize(pred[0].cpu().numpy())   # [T, N, 2]
        fut_denorm  = test_dataset.denormalize(fut[0].cpu().numpy())
        diff = pred_denorm - fut_denorm
        dist = np.linalg.norm(diff, axis=-1)  # [T, N]
        bo_total_ade += dist.mean()
        bo_total_fde += dist[-1].mean()
        bo_count += 1

        # 你原有的详细分组指标（同样走反归一化）
        metrics = compute_metrics(pred[0], fut[0], test_dataset)
        all_metrics.append(metrics)

        all_predictions.append({
            'obs': obs[0].cpu().numpy(),
            'pred': pred[0].cpu().numpy(),
            'gt': fut[0].cpu().numpy(),
            'metrics': metrics
        })

    print("\n" + "="*60)
    print("Test Results Summary (denormalized units)")
    print("="*60)

    avg_ade = np.mean([m['ade'] for m in all_metrics])
    avg_fde = np.mean([m['fde'] for m in all_metrics])
    std_ade = np.std([m['ade'] for m in all_metrics])
    std_fde = np.std([m['fde'] for m in all_metrics])

    print(f"\nOverall Performance:")
    print(f"  ADE: {avg_ade:.2f} ± {std_ade:.2f}")
    print(f"  FDE: {avg_fde:.2f} ± {std_fde:.2f}")

    print(f"\nBall Performance:")
    print(f"  ADE: {np.mean([m['ball_ade'] for m in all_metrics]):.2f}")
    print(f"  FDE: {np.mean([m['ball_fde'] for m in all_metrics]):.2f}")

    print(f"\nOffense Team Performance:")
    print(f"  ADE: {np.mean([m['offense_ade'] for m in all_metrics]):.2f}")
    print(f"  FDE: {np.mean([m['offense_fde'] for m in all_metrics]):.2f}")

    print(f"\nDefense Team Performance:")
    print(f"  ADE: {np.mean([m['defense_ade'] for m in all_metrics]):.2f}")
    print(f"  FDE: {np.mean([m['defense_fde'] for m in all_metrics]):.2f}")

    ade_list = [m['ade'] for m in all_metrics]
    best_idx = int(np.argmin(ade_list))
    worst_idx = int(np.argmax(ade_list))
    print(f"\nBest prediction:  Sample {best_idx} (ADE: {ade_list[best_idx]:.2f})")
    print(f"Worst prediction: Sample {worst_idx} (ADE: {ade_list[worst_idx]:.2f})")

    print("\nBest-of-K (per-sample, denormalized) running mean:")
    print(f"  ADE: {bo_total_ade / bo_count:.2f}")
    print(f"  FDE: {bo_total_fde / bo_count:.2f}")

    # 保存结果
    if args.save_results:
        results = {
            'predictions': all_predictions,
            'metrics': all_metrics,
            'dataset': test_dataset,
            'args': args
        }
        save_path = os.path.join(args.output_dir, str(args.pred_len),'test_results.pkl')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {save_path}")

    return all_predictions, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trajectory prediction model')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, 
                       default='./base_datasets/football',
                       help='Path to data directory')
    parser.add_argument('--obs_len', type=int, default=8,
                       help='Observation length')
    parser.add_argument('--pred_len', type=int, default=16,
                       help='Prediction length')
    parser.add_argument('--num_agents', type=int, default=23)

    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to test (-1 for all)')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save test results')
    parser.add_argument('--output_dir', type=str, default='./result',
                       help='Output directory')
    
    parser.add_argument('--num_trajectories', type=int, default=20,
                       help='Number of trajectories for Best-of-K evaluation')
    parser.add_argument('--use_best_k', action='store_true', default=True,
                       help='Use Best-of-K evaluation')
    
    
    
    args = parser.parse_args()
    
    if args.num_samples == -1:
        args.num_samples = 100000  # 测试所有样本
    
    checkpoint_path = os.path.join(args.model_path,str(args.pred_len),'best_model.pth')
    # checkpoint_path = "./checkpoints/lr0.0001_gamma0.5_bs16_step30/16/best_model.pth"
    print(checkpoint_path)
    test_model(args)
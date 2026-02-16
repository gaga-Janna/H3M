import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pickle
import argparse
import os


scale = 1.45
legend_kwargs = dict(
    fontsize=12 * scale,
    markerscale=1.2 * scale,
    handlelength=2.5 * scale,
    handletextpad=0.8 * scale,
    columnspacing=1.0 * scale,
    borderpad=0.6 * scale,
    framealpha=0.6
)



def create_football_field(ax, home_endzone_color="#6dac8e", away_endzone_color="#6dac8e"):
    """创建符合真实标准的美式足球场地"""
    field_length = 120  # 包括两个10码端区
    field_width = 53.3
    
    # 场地背景（条纹草地效果）
    for i in range(0, 12):
        color = "#a8d08e" if i % 2 == 0 else "#7aa662"
        ax.add_patch(Rectangle((i*10, 0), 10, field_width, 
                               facecolor=color, alpha=0.4, zorder=0))
    
    # 端区
    ax.add_patch(Rectangle((0, 0), 10, field_width, 
                           facecolor=home_endzone_color, alpha=0.3, zorder=0))
    ax.add_patch(Rectangle((110, 0), 10, field_width, 
                           facecolor=home_endzone_color, alpha=0.3, zorder=0))
    
    # 端区文字
    ax.text(5, field_width/2, 'END ZONE', color='white', fontsize=20, 
           ha='center', va='center', rotation=90, alpha=0.7, weight='bold')
    ax.text(115, field_width/2, 'END ZONE', color='white', fontsize=20, 
           ha='center', va='center', rotation=90, alpha=0.7, weight='bold')
    
    # 场地边界（白线）
    ax.plot([0, field_length], [0, 0], 'w-', linewidth=3, zorder=1)
    ax.plot([0, field_length], [field_width, field_width], 'w-', linewidth=3, zorder=1)
    ax.plot([0, 0], [0, field_width], 'w-', linewidth=3, zorder=1)
    ax.plot([field_length, field_length], [0, field_width], 'w-', linewidth=3, zorder=1)
    
    # 码线和码数标记（符合真实场地）
    yard_markers = {
        10: 'G',   # Goal line
        20: '10',
        30: '20',
        40: '30',
        50: '40',
        60: '50',  # 中场
        70: '40',
        80: '30',
        90: '20',
        100: '10',
        110: 'G'   # Goal line
    }
    
    for yard, label in yard_markers.items():
        # 画码线
        if label == '50':
            # 中场线稍粗一些
            ax.plot([yard, yard], [0, field_width], 'w-', alpha=0.9, linewidth=2, zorder=1)
        else:
            ax.plot([yard, yard], [0, field_width], 'w-', alpha=0.8, linewidth=1.5, zorder=1)
        
        # 码数标记
        if label != 'G':
            # 顶部和底部的码数标记，50码线正常显示
            ax.text(yard, 2, label, color='white', fontsize=20, ha='center', weight='bold')
            ax.text(yard, field_width-2, label, color='white', fontsize=20, ha='center', weight='bold')
    
    # 添加哈希标记（短横线）
    for yard in range(10, 110, 1):
        if yard % 10 != 0:  # 不在主码线上
            # 上方哈希标记
            ax.plot([yard, yard], [field_width/3 - 0.5, field_width/3 + 0.5], 
                   'w-', alpha=0.4, linewidth=0.5, zorder=1)
            # 下方哈希标记
            ax.plot([yard, yard], [2*field_width/3 - 0.5, 2*field_width/3 + 0.5], 
                   'w-', alpha=0.4, linewidth=0.5, zorder=1)
    
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_aspect('equal')
    ax.set_facecolor("#3C733C")
    # ax.set_xlabel('Field Position (yards)', fontsize=12, color='white')
    # ax.set_ylabel('Field Width (yards)', fontsize=12, color='white')
    
    # # 设置坐标轴颜色
    # ax.tick_params(colors='white')
    # ax.spines['bottom'].set_color('white')
    # ax.spines['top'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.spines['right'].set_color('white')

def draw_trajectory_line(ax, trajectory, color, is_observed=True, is_prediction=False,
                         linewidth=2, alpha=1.0, label=None, is_ball=False, 
                         prev_trajectory=None):
    """绘制轨迹线，过去的用浅色，真实用实线，预测用虚线"""
    if len(trajectory) < 2:
        return
    
    # 根据轨迹类型设置样式
    if is_observed:
        # 过去的观察轨迹 - 浅色实线
        linestyle = '-'
        line_alpha = alpha * 0.4  # 更浅的颜色
        line_color = color
    elif is_prediction:
        # 预测轨迹 - 正常颜色虚线
        linestyle = '--'
        line_alpha = alpha * 0.8
        line_color = color
    else:
        # 真实未来轨迹 - 正常颜色实线
        linestyle = '-'
        line_alpha = alpha * 0.8
        line_color = color

    
    # 绘制连接线
    x_coords = [pos[0] for pos in trajectory]
    y_coords = [pos[1] for pos in trajectory]
    
    # 如果有前一段轨迹（观察轨迹），连接最后一个点和当前轨迹的第一个点
    if prev_trajectory is not None and len(prev_trajectory) > 0 and not is_observed:
        # 绘制连接线段（从观察轨迹最后一点到未来轨迹第一点）
        connect_x = [prev_trajectory[-1][0], trajectory[0][0]]
        connect_y = [prev_trajectory[-1][1], trajectory[0][1]]
        ax.plot(connect_x, connect_y, 
               color=line_color, linestyle='-', 
               linewidth=linewidth*0.8, alpha=line_alpha*0.6, 
               solid_capstyle='round', zorder=3)
    
    ax.plot(x_coords, y_coords, 
           color=line_color, linestyle=linestyle, 
           linewidth=linewidth, alpha=line_alpha, 
           solid_capstyle='round', zorder=3, label=label)
    
    # 标记起点和终点
    if len(trajectory) > 0:
        if is_observed:
            # 起点标记
            ax.scatter(trajectory[0][0], trajectory[0][1], 
                      color=color, s=80, 
                      edgecolor='white', linewidth=2,
                      marker='o', alpha=0.9, zorder=5)
        
        # 终点标记 - 只为球添加
        if not is_observed and is_ball:
            marker = '^' if is_prediction else 's'
            edge_color = 'yellow' if is_prediction else 'white'
            marker_size = 150  # 球的标记更大
            
            ax.scatter(trajectory[-1][0], trajectory[-1][1], 
                      color=color, s=marker_size, 
                      edgecolor=edge_color, linewidth=2,
                      marker=marker, alpha=1.0, zorder=5)

# def determine_offense_team(obs):
#     """
#     根据历史轨迹中与球的距离判断哪个队是进攻方
#     找到距离球最近的玩家，如果其ID<12则前12个是offense，否则后12个是offense
#     返回True如果前12个(1-11)是offense，False如果后12个(12-22)是offense
#     """
#     # 获取球的初始位置（第一个时间步）
#     ball_pos = obs[0, 0, :]  # 第一个时间步，第0个agent（球）
    
#     # 计算所有玩家到球的距离
#     min_distance = float('inf')
#     closest_player_idx = 1
    
#     for player_idx in range(1, 23):  # 玩家索引从1到22
#         player_pos = obs[0, player_idx, :]
#         distance = np.sqrt((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)
#         if distance < min_distance:
#             min_distance = distance
#             closest_player_idx = player_idx
    
#     # 如果最近的玩家ID < 12，则前12个是offense
#     return closest_player_idx < 12


import numpy as np

def determine_offense_team(obs, k_frames=24):
    """
    根据历史轨迹中与球的距离判断哪个队是进攻方
    - 在前 k_frames 帧中，找出距离球最近的球员
    - 如果该球员大多数在前 12 个(1..11)，返回 True
    - 如果大多数在后 12 个(12..22)，返回 False
    参数:
        obs: ndarray, [T, N, 2]，N>=23, 其中 obs[:,0,:] 是球
        k_frames: 前多少帧投票 (默认 8)
    返回:
        True  -> 前 12 个是进攻
        False -> 后 12 个是进攻
    """
    T, N, D = obs.shape
    assert N >= 23 and D == 2, f"expect obs shape [T, >=23, 2], got {obs.shape}"

    k_frames = min(k_frames, T)
    votes_front, votes_back = 0, 0

    for t in range(k_frames):
        ball_pos = obs[t, 0, :]  # 球位置
        players_pos = obs[t, 1:23, :]  # 22个球员
        d = np.linalg.norm(players_pos - ball_pos, axis=1)  # [22]
        closest = np.argmin(d)  # 0..21
        if closest < 11:   # 前 11 人 (1..11)
            votes_front += 1
        else:              # 后 11 人 (12..22)
            votes_back += 1

    if votes_front == votes_back:
        # 平票时，回退到第一帧结果
        ball_pos = obs[0, 0, :]
        players_pos = obs[0, 1:23, :]
        d = np.linalg.norm(players_pos - ball_pos, axis=1)
        closest = np.argmin(d)
        return closest < 11
    else:
        return votes_front > votes_back




def visualize_comparison(predictions, dataset, sample_indices=None, save_dir='./output', 
                        viz_mode='both'):
    """
    创建改进的对比可视化
    
    Args:
        viz_mode: 'both' - 显示两个子图
                 'comparison' - 只显示预测vs真实对比图
                 'split' - 左边offense（真实+预测），右边defense（真实+预测）
    """
    
    if sample_indices is None:
        sample_indices = range(min(5, len(predictions)))
    
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in sample_indices:
        pred_data = predictions[idx]
        obs = dataset.denormalize(pred_data['obs'])
        pred = dataset.denormalize(pred_data['pred'])
        gt = dataset.denormalize(pred_data['gt'])
        
        # 判断哪个队是进攻方
        team1_is_offense = determine_offense_team(obs)
        
        # 根据模式创建图形
        if viz_mode == 'comparison':
            fig = plt.figure(figsize=(16, 12))
            # fig.patch.set_facecolor('#0a0a0a')
            ax = plt.subplot(1, 1, 1)
            axes = [ax]
            # ax.set_title('Prediction vs Ground Truth', fontsize=16, color='white', pad=15)
        else:
            fig = plt.figure(figsize=(28, 12))
            # fig.patch.set_facecolor('#0a0a0a')
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            axes = [ax1, ax2]
            
            if viz_mode == 'split':
                ax1.set_title('Offense Trajectories (Ground Truth vs Predicted)', fontsize=14, color='white', pad=15)
                ax2.set_title('Defense Trajectories (Ground Truth vs Predicted)', fontsize=14, color='white', pad=15)
            else:  # both
                ax1.set_title('Ground Truth', fontsize=14, color='white', pad=15)
                ax2.set_title('Predictions', fontsize=14, color='white', pad=15)
        
        # 为每个子图创建场地
        for ax in axes:
            create_football_field(ax)
        
        # 颜色配置 - 红色系给offense，蓝色系给defense
        colors = {
        'ball': '#FFD700',  # 金色
        'offense': ['#d62728'],
        'defense': ["#116bac"]
        }
        # colors = {
        #     'ball': '#FFD700',  # 金色
        #     'offense': ['#d62728', '#e74c3c', '#c0392b', '#a93226', '#922b21',
        #                '#ff6b6b', '#ff8787', '#ffa3a3', '#ffbfbf', '#ffdcdc', '#7b241c'],
        #     'defense': ['#1f77b4', '#2980b9', '#3498db', '#5dade2', '#85c1e2',
        #                '#2c3e50', '#34495e', '#5d6d7e', '#85929e', '#aeb6bf', '#154360']
        # }
        
        # 绘制所有智能体轨迹
        for agent_idx in range(23):
            # 根据team1_is_offense判断当前agent属于哪个队
            if agent_idx == 0:  # 球
                color = colors['ball']
                lw = 3
                is_offense = False  # 球不属于任何队
                is_defense = False
            elif agent_idx < 12:  # 前12个(1-11)
                if team1_is_offense:
                    # 前12个是offense
                    color = colors['offense'][min(agent_idx-1, len(colors['offense'])-1)]
                    is_offense = True
                    is_defense = False
                else:
                    # 前12个是defense
                    color = colors['defense'][min(agent_idx-1, len(colors['defense'])-1)]
                    is_offense = False
                    is_defense = True
                lw = 2
            else:  # 后12个(12-22)
                if team1_is_offense:
                    # 后12个是defense
                    color = colors['defense'][min(agent_idx-12, len(colors['defense'])-1)]
                    is_offense = False
                    is_defense = True
                else:
                    # 后12个是offense
                    color = colors['offense'][min(agent_idx-12, len(colors['offense'])-1)]
                    is_offense = True
                    is_defense = False
                lw = 2
            
            # 轨迹数据
            obs_traj = obs[:, agent_idx, :]
            gt_traj = gt[:, agent_idx, :]
            pred_traj = pred[:, agent_idx, :]
            
            if viz_mode == 'comparison':
                # 单图模式：显示对比，使用正确的队伍颜色
                draw_trajectory_line(axes[0], obs_traj, color, is_observed=True, 
                                   is_prediction=False, linewidth=lw, is_ball=(agent_idx==0))
                draw_trajectory_line(axes[0], gt_traj, color, is_observed=False, 
                                   is_prediction=False, linewidth=lw, is_ball=(agent_idx==0),
                                   prev_trajectory=obs_traj)
                draw_trajectory_line(axes[0], pred_traj, color, is_observed=False, 
                                   is_prediction=True, linewidth=lw, is_ball=(agent_idx==0),
                                   prev_trajectory=obs_traj)
            
            elif viz_mode == 'split':
                # 左边：只显示offense（包括球）的真实和预测轨迹
                if agent_idx == 0 or is_offense:  # 球或进攻队员
                    draw_trajectory_line(axes[0], obs_traj, color, is_observed=True, 
                                       is_prediction=False, linewidth=lw, is_ball=(agent_idx==0))
                    draw_trajectory_line(axes[0], gt_traj, color, is_observed=False, 
                                       is_prediction=False, linewidth=lw, is_ball=(agent_idx==0),
                                       prev_trajectory=obs_traj)
                    draw_trajectory_line(axes[0], pred_traj, color, is_observed=False, 
                                       is_prediction=True, linewidth=lw, is_ball=(agent_idx==0),
                                       prev_trajectory=obs_traj)
                
                # 右边：只显示defense的真实和预测轨迹
                if is_defense:  # 防守队员
                    draw_trajectory_line(axes[1], obs_traj, color, is_observed=True, 
                                       is_prediction=False, linewidth=lw, is_ball=(agent_idx==0))
                    draw_trajectory_line(axes[1], gt_traj, color, is_observed=False, 
                                       is_prediction=False, linewidth=lw, is_ball=(agent_idx==0),
                                       prev_trajectory=obs_traj)
                    draw_trajectory_line(axes[1], pred_traj, color, is_observed=False, 
                                       is_prediction=True, linewidth=lw, is_ball=(agent_idx==0),
                                       prev_trajectory=obs_traj)
            
            else:  # both
                # 左图：Ground Truth，使用正确的队伍颜色
                draw_trajectory_line(axes[0], obs_traj, color, is_observed=True, 
                                   is_prediction=False, linewidth=lw, is_ball=(agent_idx==0))
                draw_trajectory_line(axes[0], gt_traj, color, is_observed=False, 
                                   is_prediction=False, linewidth=lw, is_ball=(agent_idx==0),
                                   prev_trajectory=obs_traj)
                
                # 右图：只显示预测（不显示真实未来轨迹），使用正确的队伍颜色
                draw_trajectory_line(axes[1], obs_traj, color, is_observed=True, 
                                   is_prediction=False, linewidth=lw, is_ball=(agent_idx==0))
                draw_trajectory_line(axes[1], pred_traj, color, is_observed=False, 
                                   is_prediction=True, linewidth=lw, is_ball=(agent_idx==0),
                                   prev_trajectory=obs_traj)
        
        # 图例
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, linestyle='-', 
                  label='Observed Trajectory (Past)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='-', 
                  label='Ground Truth (Future)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', 
                  label='Predicted (Future)'),
            Line2D([0], [0], marker='o', color='w', markersize=10, 
                  markerfacecolor=colors['ball'], label='Ball', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markersize=10, 
                  markerfacecolor='#d62728', label='Offense', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markersize=10, 
                  markerfacecolor='#1f77b4', label='Defense', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markersize=10, 
                  markerfacecolor='None', markeredgecolor='white', 
                  label='Start Position', linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markersize=10, 
                  markerfacecolor='#FFD700', markeredgecolor='yellow', 
                  label='Predicted Ball End', linestyle='None'),
            Line2D([0], [0], marker='s', color='w', markersize=10, 
                  markerfacecolor='#FFD700', markeredgecolor='white', 
                  label='True Ball End', linestyle='None')
        ]
        
        # # 根据模式调整图例
        # if viz_mode == 'both':
        #     # 左图图例
        #     left_legend = legend_elements[:6] + [legend_elements[6], legend_elements[8]]
        #     axes[0].legend(handles=left_legend, loc='upper right', 
        #                   ncol=2)
            
        #     # 右图图例（不包含Ground Truth）
        #     right_legend = [legend_elements[0], legend_elements[2]] + legend_elements[3:8]
        #     axes[1].legend(handles=right_legend, loc='upper right', 
        #                  ncol=2)
        # elif viz_mode == 'split':
        #     # 左图图例（offense相关）
        #     left_legend = legend_elements[:4] + [legend_elements[4], legend_elements[6], legend_elements[7], legend_elements[8]]
        #     axes[0].legend(handles=left_legend, loc='upper right', 
        #                   ncol=2)
            
        #     # 右图图例（defense相关）
        #     right_legend = legend_elements[:3] + [legend_elements[5], legend_elements[6]]
        #     axes[1].legend(handles=right_legend, loc='upper right', 
        #                  ncol=2)
        # else:
        #     for ax in axes:
        #         ax.legend(handles=legend_elements, loc='upper left',
        #                  ncol=2, **legend_kwargs)
        
        plt.tight_layout(pad=0)
        


        save_path = os.path.join(save_dir, f'trajectory_{viz_mode}_{idx}.pdf')
        plt.savefig(save_path, facecolor=fig.get_facecolor(), 
            edgecolor='none', bbox_inches='tight')
    
        # save_path = os.path.join(save_dir, f'trajectory_{viz_mode}_{idx}.png')
        # plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), 
        #         edgecolor='none', bbox_inches='tight')
        plt.close()

        
        print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory predictions')
    parser.add_argument('--results_path', type=str, 
                       default='./result/16/test_results.pkl',
                       help='Path to test results file')
    parser.add_argument('--num_visualize', type=int, default=150,
                       help='Number of samples to visualize')
    parser.add_argument('--sample_indices', nargs='+', type=int,
                       help='Specific sample indices to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations_football',
                       help='Output directory for visualizations')
    parser.add_argument('--viz_mode', type=str, default='comparison',
                       choices=['both', 'comparison', 'split'],
                       help='Visualization mode: both, comparison, or split')
    
    args = parser.parse_args()

    # 加载测试结果
    with open(args.results_path, 'rb') as f:
        results = pickle.load(f)
    
    predictions = results['predictions']
    dataset = results['dataset']
    
    # 确定要可视化的样本
    if args.sample_indices:
        sample_indices = args.sample_indices
    else:
        sample_indices = range(min(args.num_visualize, len(predictions)))
    
    # 创建可视化
    visualize_comparison(predictions, dataset, sample_indices, 
                        args.output_dir, args.viz_mode)
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")
    print(f"Visualization mode: {args.viz_mode}")

if __name__ == '__main__':
    main()



import itertools
import subprocess
import os
import re
import sys

# ========== 1. 搜索空间定义 ==========
lrs = [0.0001, 0.0005, 0.001]       # 学习率
gammas = [0.5]             # 衰减系数
batch_sizes = [16, 32, 64]         # 批大小
step_sizes = [30]            # 衰减步长

# 保存日志和结果的目录
log_dir = "grid_search_logs"
os.makedirs(log_dir, exist_ok=True)
result_file = "grid_search_results.txt"

# 检查train.py是否存在
if not os.path.exists("train.py"):
    print("[ERROR] train.py not found in current directory!")
    sys.exit(1)

# ========== 2. 运行单次实验 ==========
def run_experiment(lr, gamma, batch_size, step_size):
    """
    运行单个实验，实时监控输出。
    返回：(success_flag, final_ADE or None)
    """
    exp_name = f"lr{lr}_gamma{gamma}_bs{batch_size}_step{step_size}"
    log_path = os.path.join(log_dir, f"{exp_name}.log")
    print(f"\n===== Running {exp_name} =====")

    cmd = [
        "python", "train.py",
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--num_epochs", "100",
        "--decay_step", str(step_size),
        "--decay_gamma", str(gamma),
        "--save_path", f"./checkpoints/{exp_name}",
        "--no_tqdm"  # <--- 新增
    ]

    nan_detected = False
    final_ade = None

    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")            # 打印实时输出
                log_file.write(line)
                log_file.flush()

                # 检测loss为NaN
                if "nan" in line.lower():
                    print(f"[WARNING] NaN loss detected in {exp_name}, stopping this run...")
                    nan_detected = True
                    process.kill()
                    break

                # 提取ADE值（假设train.py在评估时输出类似： "Test - ADE: 1.234, FDE: 2.345"）
                match = re.search(r"ADE:\s*([0-9]*\.?[0-9]+)", line)
                if match:
                    final_ade = float(match.group(1))

            process.wait()

    except Exception as e:
        print(f"[ERROR] Exception while running {exp_name}: {str(e)}")
        return False, None

    # 检查运行结果
    if nan_detected:
        return False, None
    elif process.returncode != 0:
        print(f"[ERROR] {exp_name} crashed with code {process.returncode}.")
        return False, None
    else:
        return True, final_ade


# ========== 3. 主循环：遍历所有组合 ==========
results = []  # 保存每次实验的记录

with open(result_file, "w") as rf:
    rf.write("Grid Search Results\n")
    rf.write("===================\n")

    for lr, gamma, batch_size, step_size in itertools.product(lrs, gammas, batch_sizes, step_sizes):
        success, ade = run_experiment(lr, gamma, batch_size, step_size)

        exp_name = f"lr{lr}_gamma{gamma}_bs{batch_size}_step{step_size}"
        if success and ade is not None:
            rf.write(f"{exp_name} | Final ADE: {ade:.4f}\n")
            results.append((ade, {"lr": lr, "gamma": gamma,
                                  "batch_size": batch_size, "step_size": step_size}))
        elif not success:
            rf.write(f"{exp_name} | FAILED\n")

# ========== 4. 输出最优配置 ==========
if results:
    results.sort(key=lambda x: x[0])  # 按ADE升序排序
    best_ade, best_config = results[0]
    print("\n===== Grid Search Completed =====")
    print(f"Best Configuration: lr={best_config['lr']}, gamma={best_config['gamma']}, "
          f"batch_size={best_config['batch_size']}, step_size={best_config['step_size']}")
    print(f"Best ADE: {best_ade:.4f}")

    with open(result_file, "a") as rf:
        rf.write("\n===== Best Result =====\n")
        rf.write(f"Best Config: {best_config}\n")
        rf.write(f"Best ADE: {best_ade:.4f}\n")
else:
    print("[WARNING] No successful runs. Please check logs for details.")

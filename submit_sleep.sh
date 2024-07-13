#!/bin/bash
#SBATCH --job-name=llava_Cambrian   # 作业名称
#SBATCH --output=slurm_logs/output_%j.log       # 标准输出和错误日志文件名（%j为作业ID）
#SBATCH --ntasks=1                   # 启动的进程数量
#SBATCH --nodes=1                    # 请求的节点数量
#SBATCH --partition=batch          # 作业提交到的分区
#SBATCH --gres=gpu:8


# 加载必要的模块 (如有需要)
# module load python/3.8

# 打印一些信息以供调试
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated cores: $SLURM_NTASKS"
echo "Job started at: $(date)"

# 运行Python脚本
python sleep.py

# 打印作业结束时间
echo "Job finished at: $(date)"

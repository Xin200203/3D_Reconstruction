#!/bin/bash
# GPU利用率持续监控脚本
echo "开始GPU监控..."
echo "时间,GPU利用率%,内存使用MB,内存总量MB,内存利用率%" > gpu_usage.csv

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    echo "$timestamp,$gpu_info" >> gpu_usage.csv
    echo "$timestamp: $gpu_info"
    sleep 10
done

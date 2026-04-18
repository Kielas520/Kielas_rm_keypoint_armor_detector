# 方案 A：如果官方镜像拉取依然失败，建议在本地 pull 成功后再 build，
# 或者临时更换国内可用的代理仓库地址（如：docker.pullmirror.com/pytorch/pytorch:...）
# 使用自带 CUDA 和 PyTorch 的官方镜像
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 环境变量优化
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 更新源并安装 OpenCV 必备的系统依赖（补充了渲染和扩展库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 复制配置文件
COPY pyproject.toml .

# 优化 pip 安装：使用清华源加速依赖下载（如果是在国内构建）
# 同时安装可选的 [dev] 或 [train] 额外依赖（如果有的话）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple .

# 默认运行命令
CMD ["python", "main.py"]


import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
import math

class FeatureExtractor:
    """
    基于 PyTorch Forward Hook 的特征提取器
    """
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        # 遍历模型挂载 hook
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))
                
    def _get_hook(self, name):
        def hook(model, input, output):
            # 将对应层的输出特征图分离并转移到 CPU
            self.features[name] = output.detach().cpu()
        return hook
        
    def remove_hooks(self):
        """用完后及时清理 hook，防止内存泄漏"""
        for hook in self.hooks:
            hook.remove()

def save_feature_map_grid(feature_tensor, save_path, layer_name, max_channels=64):
    """
    将特征图按通道绘制成网格图并保存
    feature_tensor: [C, H, W] 形状的特征图张量
    max_channels: 最大展示通道数，防止特征图过多导致图片过大（默认取前64个通道）
    """
    num_channels = feature_tensor.shape[0]
    plot_channels = min(num_channels, max_channels)
    
    # 计算网格布局，例如 64 通道 -> 8x8 网格
    grid_size = math.ceil(math.sqrt(plot_channels))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
    fig.suptitle(f"Layer: {layer_name} (First {plot_channels} Channels)", fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        ax = axes[i]
        ax.axis('off')
        
        if i < plot_channels:
            f_map = feature_tensor[i].numpy()
            
            # 检查特征图是否全为 0 (神经元坏死)
            f_min, f_max = f_map.min(), f_map.max()
            if f_min == f_max:
                # 恒定值特征图（坏死或饱和），直接用全黑/灰色表示
                ax.imshow(np.zeros_like(f_map), cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f"CH: {i} (Dead)", color='red', fontsize=8)
            else:
                # 归一化并绘制
                normalized_map = (f_map - f_min) / (f_max - f_min + 1e-8)
                ax.imshow(normalized_map, cmap='viridis')
                ax.set_title(f"CH: {i}", fontsize=8)
                
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

@torch.no_grad()
def visualize_predictions_with_features(model, dataloader, device, save_dir, prefix, progress, input_size, strides, reg_max, process_multi_scale_dets_fn, num_samples=5, conf_threshold=0.5, kpt_dist_thresh=14.5):
    """
    集成了特征通道可视化的推理函数，将替换 train.py 中的 visualize_predictions
    注意：通过参数 process_multi_scale_dets_fn 传入原 train.py 中的解码函数，避免循环导入
    """
    model.eval()
    
    # 定义你想要观察的层。针对 RMDetector，通常观察 Neck 的输出层
    # 你可以通过 print(model) 查看具体的层名，这里预设监控 PAN 结构输出的 P3, P4, P5
    target_layers = [
        'neck.conv_f3', 
        'neck.conv_p4', 
        'neck.conv_p5'
    ]
    
    extractor = FeatureExtractor(model, target_layers)
    count = 0
    
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像与特征图...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        # 前向传播，extractor 会自动截获 target_layers 的输出
        preds = model(imgs) 
        
        # 使用原有的解码逻辑
        gt_dets, pred_dets = process_multi_scale_dets_fn(
            preds, targets, class_ids, strides, input_size, 
            reg_max, conf_threshold, kpt_dist_thresh
        )
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                extractor.remove_hooks()
                progress.remove_task(task_id)
                return
            
            # --- 1. 创建该 Sample 的专属文件夹 ---
            sample_dir = Path(save_dir) / f"{prefix}_sample_{count+1}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 2. 绘制并保存主推理对比图 (Result) ---
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_np)
            
            # 画 GT
            if len(gt_dets[i]) > 0:
                for det in gt_dets[i]:
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    ax.scatter(pts[:, 0], pts[:, 1], color='lime', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='lime', linewidth=2)
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='lime', linewidth=2)
                    ax.annotate(f"GT ID: {cls_id}", xy=(cx, cy), xytext=(cx - 60, cy - 40), color='lime', weight='bold')
            
            # 画 Pred
            if len(pred_dets[i]) > 0:
                for det in pred_dets[i]:
                    score = det[0]
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    ax.scatter(pts[:, 0], pts[:, 1], color='red', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='red', linewidth=2, linestyle='--')
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='red', linewidth=2, linestyle='--')
                    ax.annotate(f"Pred ID: {cls_id}", xy=(cx, cy), xytext=(cx + 60, cy - 40), color='red', weight='bold')
                    ax.annotate(f"Conf: {score:.2f}", xy=(cx, cy), xytext=(cx + 60, cy + 40), color='red', weight='bold')
            
            plt.title(f"{prefix} Set - Sample {count+1}\nGreen: GT | Red: Pred")
            plt.axis('off')
            plt.savefig(sample_dir / "inference_result.png", bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            # --- 3. 绘制并保存特征通道图 ---
            for layer_name in target_layers:
                if layer_name in extractor.features:
                    # 获取当前 batch 中第 i 张图片的特征图: shape [C, H, W]
                    feature_tensor = extractor.features[layer_name][i]
                    save_path = sample_dir / f"features_{layer_name.replace('.', '_')}.png"
                    save_feature_map_grid(feature_tensor, save_path, layer_name, max_channels=64)
            
            count += 1
            progress.update(task_id, advance=1)
            
    extractor.remove_hooks()
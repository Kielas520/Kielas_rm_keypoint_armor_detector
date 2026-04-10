import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.console import Console

# 导入我们之前写好的模块
from src.datasets import RMArmorDataset
from src.model import RMDetector
from src.loss import RMDetLoss

console = Console()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        preds = model(imgs)
        
        # 3. 计算损失
        loss, loss_dict = criterion(preds, targets)
        
        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印日志 (每 10 个 Batch 打印一次)
        if batch_idx % 10 == 0:
            console.print(f"[Train] Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f} (Conf: {loss_dict['loss_conf']:.4f}, "
                          f"Box: {loss_dict['loss_box']:.4f}, Pose: {loss_dict['loss_pose']:.4f})")
            
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # -----------------------------------------
    # 1. 超参数与配置
    # -----------------------------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 临时修改为强制 CPU：
    device = torch.device("cpu")
    console.print(f"[bold green]Using device: {device}[/bold green]")
    
    batch_size = 4
    epochs = 1
    learning_rate = 1e-3
    save_dir = "./weights"
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------------------
    # 2. 初始化 Dataset 和 DataLoader
    # -----------------------------------------
    # 注意：这里的路径需要根据你 split.py 生成的实际路径进行调整
    train_dataset = RMArmorDataset(
        img_dir="./data/datasets/images/train", 
        label_dir="./data/datasets/labels/train"
    )
    val_dataset = RMArmorDataset(
        img_dir="./data/datasets/images/val", 
        label_dir="./data/datasets/labels/val"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # -----------------------------------------
    # 3. 初始化 模型、损失函数和优化器
    # -----------------------------------------
    model = RMDetector().to(device)
    
    # 将 yaml 中的类别权重传入损失函数 (如果需要做分类损失的话)
    # 此处按你的要求先处理目标检测与关键点联合估计的 Loss
    criterion = RMDetLoss(lambda_conf=1.0, lambda_box=2.0, lambda_pose=1.0).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 可选：学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -----------------------------------------
    # 4. 主训练循环
    # -----------------------------------------
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        console.rule(f"Epoch {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        console.print(f"[bold cyan]Epoch {epoch} Summary:[/bold cyan] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            console.print(f"[bold red]Best model saved to {save_path}[/bold red]")
            
    # 保存最后一个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    console.print("[bold green]Training Completed![/bold green]")

if __name__ == "__main__":
    main()
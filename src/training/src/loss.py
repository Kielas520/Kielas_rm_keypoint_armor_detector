import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss

class RMDetLoss(nn.Module):
    def __init__(self, lambda_conf=1.0, lambda_box=2.0, lambda_pose=1.0, grid_size=(10, 10)):
        """
        联合损失函数模块
        lambda_conf, lambda_box, lambda_pose 分别为各项损失的权重系数
        """
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_box = lambda_box
        self.lambda_pose = lambda_pose
        self.grid_w, self.grid_h = grid_size
        
        # 使用 BCEWithLogitsLoss 计算置信度损失，内部自带 Sigmoid，数值稳定性更佳
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        # 关键点像素偏移使用 Smooth L1 损失，对抗异常点具有较好鲁棒性
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def _decode_boxes(self, boxes, grid_y, grid_x):
        """
        解码边界框坐标：
        将网络输出的相对网格偏移量 (tx, ty, w, h) 解码为全图归一化的 (x1, y1, x2, y2)
        """
        tx, ty = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2], boxes[:, 3]
        
        # 还原为全图范围内的归一化中心点坐标 [0, 1]
        cx = (tx + grid_x) / self.grid_w
        cy = (ty + grid_y) / self.grid_h
        
        # 计算左上角和右下角的绝对归一化坐标
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, pred, target):
        """
        前向传播计算损失
        pred:   [Batch, 13, H_grid, W_grid]
        target: [Batch, 13, H_grid, W_grid]
        """
        # -----------------------------------------
        # 1. 置信度损失 (全局所有网格均参与计算)
        # -----------------------------------------
        pred_conf = pred[:, 0, :, :]
        target_conf = target[:, 0, :, :]
        loss_conf = self.bce(pred_conf, target_conf)

        # -----------------------------------------
        # 2. 提取正样本 (制作 Mask 掩码)
        # 仅对存在装甲板目标的网格计算 Box 和 Pose 损失
        # -----------------------------------------
        pos_mask = (target_conf == 1.0)
        
        # 若当前批次输入没有任何目标，直接返回置信度损失，避免索引报错
        if not pos_mask.any():
            return loss_conf * self.lambda_conf, {
                'loss_conf': loss_conf.item(),
                'loss_box': 0.0,
                'loss_pose': 0.0,
                'total_loss': (loss_conf * self.lambda_conf).item()
            }
        
        # 获取正样本网格在张量中的确切坐标位置
        indices = torch.nonzero(pos_mask)
        grid_y = indices[:, 1]  # H_grid 的索引
        grid_x = indices[:, 2]  # W_grid 的索引

        # -----------------------------------------
        # 3. 边界框损失 (CIoU Loss)
        # -----------------------------------------
        # 提取正样本对应的框参数，并调整形状为 [N, 4]
        pred_boxes_raw = pred[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_boxes_raw = target[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]

        # 调用解码方法
        pred_boxes = self._decode_boxes(pred_boxes_raw, grid_y, grid_x)
        target_boxes = self._decode_boxes(target_boxes_raw, grid_y, grid_x)
        
        # 调用 torchvision 原生 CIoU 函数计算损失
        loss_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')

        # -----------------------------------------
        # 4. 关键点损失 (Smooth L1 Loss)
        # -----------------------------------------
        # 提取正样本对应的 8 维关键点偏移量，形状为 [N, 8]
        pred_pose = pred[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_pose = target[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        loss_pose = self.smooth_l1(pred_pose, target_pose)

        # -----------------------------------------
        # 5. 计算带权重的总损失
        # -----------------------------------------
        total_loss = (
            self.lambda_conf * loss_conf + 
            self.lambda_box * loss_box + 
            self.lambda_pose * loss_pose
        )

        # 组装为字典，供训练脚本在控制台或 TensorBoard 中监控各个维度的下降情况
        loss_dict = {
            'loss_conf': loss_conf.item(),
            'loss_box': loss_box.item(),
            'loss_pose': loss_pose.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict
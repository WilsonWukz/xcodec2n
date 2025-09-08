import torch
import torch.optim as optim

def WarmupLR(optimizer, warmup_step=0,  down_step=5e4, max_lr=1e-4, min_lr=1e-5, **kwargs):
    base_lr = optimizer.defaults['lr']  # 初始lr（来自config）
    # alpha = (max_lr - 1e-5) / warmup_step**2
    alpha = (max_lr - min_lr) / warmup_step**2 if warmup_step > 0 else 0

    def lr_lambda(step):
        if step < warmup_step:  # warmup阶段 (二次曲线增长)
            lr = min_lr + alpha * step**2
        elif step < warmup_step + down_step:  # 衰减阶段 (线性下降)
            s1, s2 = warmup_step, warmup_step + down_step
            lr = (max_lr - min_lr) / (s1 - s2) * step + (min_lr*s1 - max_lr*s2) / (s1 - s2)
        else:  # 保持最小学习率
            lr = min_lr
        return lr / base_lr  # ⚠️ 关键：返回比例，而不是绝对lr
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # def lr_lambda(step):
    #     init_lr = 1e-5
    #     s1, s2 = warmup_step, warmup_step + down_step
    #     if step < s1:
    #         return init_lr + alpha * step**2
    #     elif s1 <= step < s2:
    #         return (max_lr - min_lr) / (s1 - s2) * step + (min_lr*s1 - max_lr*s2) / (s1 - s2)
    #     else:
    #         return min_lr
    # return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

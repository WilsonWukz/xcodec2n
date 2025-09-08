import wandb
import pandas as pd

# 登录
api = wandb.Api()

# 获取一个 run
run = api.run("username/project_name/run_id")

# run.history() 获取完整的 epoch-wise 数据
history = run.history()

# 保存为 Excel
history.to_excel("wandb_history.xlsx", index=False)

# 也可以保存到 CSV
history.to_csv("wandb_history.csv", index=False)

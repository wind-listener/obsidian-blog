---
aliases:
  - Wandb
  - wandb
---
使用 **Weights and Biases (wandb)** 进行超参数调优是一个高效且系统化的方式，可以帮助你自动记录实验的设置和结果，生成可视化图表，并轻松地比较不同的实验结果。下面是如何使用 wandb 来高效调整超参数的具体步骤：

  

**1. 安装 wandb**

  

首先需要安装 wandb 库：

```
pip install wandb
```

**2. 初始化 wandb**

  

在你的代码中，首先需要导入并初始化 wandb：

```
import wandb

# 初始化wandb，项目名称可以自定义
wandb.init(project="your_project_name", config={
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
})
```

wandb.init() 会自动为你的实验创建一个唯一的运行ID，并记录你的配置。你可以在 config 字典中设置超参数的初始值。

  

**3. 在代码中记录超参数和性能指标**

  

为了有效地追踪实验过程中的超参数和性能，使用 wandb.log() 来记录你每个 epoch 或者每个训练步骤的指标（例如：损失、准确率等）。

```
# 假设你的训练循环如下
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss, val_acc = validate_model()

    # 在每个epoch结束后记录指标
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })
```

**4. 超参数搜索（Hyperparameter Sweeps）**

  

wandb 提供了 **超参数搜索** 功能，可以自动调整和优化你的超参数。你只需要定义超参数空间，并让 wandb 在多个超参数组合上进行实验。这样可以节省大量的人工调参时间。

  

**4.1 定义超参数空间**

  

在 wandb 中定义超参数空间时，可以使用 wandb.sweep() 功能。首先在你的脚本中，修改超参数部分，使其能够接受外部传入的参数。

  

例如：

```
import wandb

# 初始化wandb
wandb.init()

# 从wandb获取超参数配置
config = wandb.config

# 使用wandb.config设置的超参数
batch_size = config.batch_size
learning_rate = config.learning_rate

# 训练模型
for epoch in range(config.epochs):
    train_loss = train_one_epoch(batch_size, learning_rate)
    val_loss, val_acc = validate_model()
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })
```

**4.2 创建 Sweep 配置**

  

在定义了超参数后，接下来你需要定义一个 Sweep 配置，指定超参数的搜索空间。

```
sweep_configuration = {
    'method': 'random',  # 'grid'，'random'，'bayes'（贝叶斯优化）
    'name': 'sweep_run_01',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'  # 目标是最大化val_accuracy
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-1
        },
        'epochs': {
            'value': 10
        }
    }
}
```

**4.3 启动 Sweep**

  

一旦你定义好配置，就可以启动 Sweep 了。wandb.sweep() 会自动运行多个实验，尝试不同的超参数组合。

```
# 创建 Sweep
sweep_id = wandb.sweep(sweep_configuration, project="your_project_name")

# 启动 Sweep
wandb.agent(sweep_id, function=train, count=10)  # count=10表示尝试10次不同的配置
```

在这个过程中，wandb 会自动记录每次实验的超参数和结果，并且在 Web 界面中为你提供详细的可视化和比较功能。

  

**5. 分析和可视化实验结果**

  

一旦超参数搜索结束，wandb 会在 Web 页面上自动汇总结果，显示每个实验的超参数与性能指标。你可以通过以下方式分析：

• **Sweep Overview**: 查看不同超参数组合的性能。

• **Hyperparameter Importance**: 了解哪些超参数对模型性能影响最大。

• **Parallel Coordinates Plot**: 比较不同超参数组合的多维度性能。

  

通过这些可视化图表，你可以更容易地找到最佳的超参数组合。

  

**6. 使用早停（Early Stopping）**

  

你还可以使用 wandb 来实现早停机制，自动停止性能不佳的实验。你只需要将 wandb 的 early_stopping 参数传递到训练函数。

```
wandb.init(config=config, project="your_project_name", early_stopping_metric="val_loss", early_stopping_threshold=0.01)
```

**总结**

1. **初始化 wandb**，在代码中记录超参数和性能指标。

2. **定义超参数搜索空间**，通过 wandb.sweep() 进行自动化超参数调优。

3. **使用 wandb.agent()** 启动多个实验，记录不同超参数组合的结果。

4. **在 Web 界面中可视化和比较实验结果**，通过图表分析最佳的超参数组合。

5. 可以结合 **早停机制**，避免无效的实验浪费计算资源。

  

通过这些步骤，你可以高效地使用 wandb 进行超参数调优，节省大量手动调参的时间，提高模型的性能。
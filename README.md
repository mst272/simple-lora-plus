# 🚀 Simple lora plus 🚀
A simple implementation of LoRA+: Efficient Low Rank Adaptation of Large Models
## 介绍
对lora+论文进行了简单的实现，并以微调deepseek-coder为例，将实现的lora+方法应用在deepseek-coder微调中🎉

在这里只是以deepseek为例，使用本项目构建好的的lora+方法你也可以对其他模型进行微调。

## 目录结构

**fintune**：此目录下是基于deepseek-coder官方实现的微调代码进行修改以适用的微调脚本。

**tricks**：目录下lora_plus.py即为lora+的实现代码

## 使用&细节

---
### 环境要求
因为是以deepseek-coder为例进行实验，所以环境要求一致。

### lora+ 使用
只需要将你的其他训练脚本中（需要是lora训练）的Trainer改为LoraPlusTrainer即可使用lora+进行训练！👋
```python
#原始
# trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
# 加入lora+ Trainer
trainer = LoraPlusTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
```

基于huggingface的Trainer，进行LoraPlusTrainer的编写，并且重写create_optimizer方法。
详细代码如下，

```python
class LoraPlusTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            lora_lr_ratio = LORA_LR_RATIO
            lora_lr_embedding = LORA_LR_EMBEDDING

            self.optimizer = create_lorap_optimizer(opt_model, lora_lr_ratio, optimizer_cls, optimizer_kwargs,
                                                    lora_lr_embedding)
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
```

### deepseek-coder例子
运行如下命令即可

值得注意的是ds_config_zero3.json文件与原始有所不同，去除了学习率的相关参数，因为lora+的实现简单讲就是调整lora中A和B学习率

> bash run_lora.sh


## 贡献指南
🤝 如果你有任何改进建议、发现了bug或者想要添加新功能，请随时提交issue或pull request。




### 安装 
```bash
pip install xspike
```

### 安装 Redis
在终端中输入以下命令后，根据提示安装 Redis 并启动 Redis 数据维护服务：
```bash
xspike
```


### GPU 自动选择、排队
```python
import xspike as x

queuer = x.GPUQueuer()

# Start the queuer
queuer.start()

# Your code here

# Stop the queuer
queuer.close()
```


### 批量启动实验
将实验启动命令放在项目目录下的 exp_plans/exp_demo/xxx.sh 文件中，每条命令之间用换行分割，示例如下：
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    'lr=5e-2' \
    'max_epochs=1' \
    'lora_rank=512' \
    'lora_alpha=1024' 

CUDA_VISIBLE_DEVICES=0 python run.py \
    'lr=5e-3' \
    'max_epochs=1' \
    'lora_rank=1024' \
    'lora_alpha=2018' 

# 其他实验命令......

```
并在项目目录下运行以下命令：
```bash
xx
```


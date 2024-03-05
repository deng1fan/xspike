
完整版介绍可以看 [xspike——GPU 任务排队、一键批量启动实验......](https://zhuanlan.zhihu.com/p/685132608/preview?comment=0&catalog=0)


使用以下命令安装 xspike：
```
pip install xspike
```

xspike 目前可支持的功能有：

- [x] GPU 任务排队 （基于 Redis）
- [x] 批量启动实验脚本
- [x] 钉钉通知
- [x] Comet.ml 管理

## 安装 Redis、启动 Redis、Redis 数据维护
在命令行输入
```
xspike
```

按照提示依次执行 4、5、6 号操作即可部署好 Redis 环境



## 启动 GPU 任务排队

```
import xspike as x


queuer = x.GPUQueuer()
queuer.start()

# Your code is here
# ......

queuer.close()
```

建议在原有的代码最前端插入上述代码，并在实验结束后调用 queuer.close() 来释放当前任务

GPUQueuer() 的初始化参数说明：

```
visible_cuda (str, optional): 可见的 GPU 编号，多个 GPU 编号用逗号分隔，如 "0,1,2,3". 默认为 "-1"，即全部可见.
n_gpus (int, optional): 需要的 GPU 数量. 默认为 1.
memo (str, optional): 任务备注. 默认为 "no memo".
```

## 实验计划
即插即用，对原来代码无任何影响

只需要在代码根目录下创建一个名为 “exp_plans” 的文件夹，里边存放每个实验计划要运行的所有脚本，并以 “xxx.sh” 来命名，样例可前往知乎博客查看


如果要取消某个实验，直接注释即可，批量启动时，为了避免 GPU 竞争，设置了实验启动间隔，每分钟启动一个实验

启动时，只需要在终端输入：
```
xx
```

即可自动扫描当前目录下的所有计划文件，并在选择后进行启动




## 钉钉通知
集成了 DingtalkChatbot 的通知功能，设置好 DINGDING_ACCESS_TOKEN 和 DINGDING_SECRET 环境变量或直接在函数调用时通过参数传递，然后调用下面的代码进行通知：

```
x.notice("Hello World!")
```

## Comet.ml 实验管理
目前这里只实现了 Comet 环境的创建和文件夹的上传

```
comet_client = CometClient(project_name="Default", api_key="xxx", exp_name="MoE Baseline")
```

上传文件夹，比如上传本次实验的代码，方便结果复现
将文件夹下的所有 ".py", ".yml" 文件进行上传（包括子文件夹）

```
comet_client.log_directory("./")
```

## 依赖
xspike 非常轻量化，依赖包很少，可以减少与现有的环境发生依赖冲突的情况
```
 nvitop
 redis
 rich
 psutil
 jsonlines
 setproctitle
 dingtalkchatbot
 comet_ml
```
## 未来计划
- [ ] 集成 Comet.ml 实验记录，使其通过最少的代码可以用在各大框架中，并提供便捷的记录模板（Callback）
- [ ] 尽量在不增加依赖的情况下，集成实验中常用的工具类和方法

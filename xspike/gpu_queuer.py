from nvitop import select_devices, Device
import time
import os
from xspike.redis_client import RedisClient
import datetime
from loguru import logger


class GPUQueuer:
    def __init__(self, visible_cuda="-1", n_gpus=1, memo="no memo"):
        """初始化GPUQueuer类

        Args:
            visible_cuda (str, optional): 可见的 GPU 编号，多个 GPU 编号用逗号分隔，如 "0,1,2,3". Defaults to "-1".
            n_gpus (int, optional): 需要的 GPU 数量. Defaults to 1.
            memo (str, optional): 任务备注. Defaults to "no memo".
        """
        self.visible_cuda = visible_cuda
        if visible_cuda == "-1" and os.environ.get("CUDA_VISIBLE_DEVICES"):
            self.visible_cuda = str(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.n_gpus = n_gpus
        self.memo = memo
        self.redis_client = RedisClient()
        self.devices = Device.all()
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        self.id = str(os.getpid()) + str(
            int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S")))
        )

    @logger.catch
    def start(self):
        # ---------------------------------------------------------------------------- #
        #                         获取当前符合条件的所有处理器
        # ---------------------------------------------------------------------------- #
        redis_client = RedisClient()
        self_occupied_gpus = redis_client.get_self_occupied_gpus()
        devices = Device.all()
        if self.visible_cuda != "-1":
            devices = [
                Device(index=int(device_id))
                for device_id in self.visible_cuda.split(",")
            ]

        devices = [
            device for device in devices if device.index not in self_occupied_gpus
        ]

        if len(devices) >= self.n_gpus:
            cuda = select_devices(
                devices=devices,
                format="index",
                min_count=self.n_gpus,
            )
            cuda = [str(x) for x in cuda]
            cuda = ",".join(cuda)
            self.cuda = cuda
            redis_client.register_process(
                self.id, cuda, n_gpus=self.n_gpus, memo=self.memo
            )
            logger.info(f"获取到足够的卡，当前分配的卡为：{cuda}")
            return cuda
        else:
            # ---------------------------------------------------------------------------- #
            #                         如果需要排队就送入队列
            # ---------------------------------------------------------------------------- #
            wait_num = redis_client.join_wait_queue(
                self.id, n_gpus=self.n_gpus, memo=self.memo
            )

        # ---------------------------------------------------------------------------- #
        #                         排队模式，等待处理器
        # ---------------------------------------------------------------------------- #
        wait_count = 0
        while not redis_client.is_my_turn(self.id) or not is_processing_units_ready(
            devices, self.n_gpus
        ):
            time.sleep(30)
            curr_time = str(time.strftime("%m月%d日 %H:%M:%S", time.localtime()))
            if redis_client.is_my_turn(self.id):
                # 更新队列
                redis_client.update_queue(self.id)

            wait_num = len(redis_client.client.lrange("wait_queue", 0, -1)) - 1
            print(
                f"\r更新时间: {curr_time} | 该任务（PID：{os.getpid()}）需要 {self.n_gpus} 块卡，前面还有 {wait_num} 个排队任务，已刷新 {wait_count} 次",
                end="",
                flush=True,
            )

            self_occupied_gpus = redis_client.get_self_occupied_gpus()

            devices = Device.all()
            if self.visible_cuda != "-1":
                devices = [
                    Device(index=int(device_id))
                    for device_id in self.visible_cuda.split(",")
                ]
            devices = [
                device for device in devices if device.index not in self_occupied_gpus
            ]
            wait_count += 1

        # ---------------------------------------------------------------------------- #
        #                         更新可用处理器
        # ---------------------------------------------------------------------------- #
        cuda = select_devices(
            devices=devices,
            format="index",
            min_count=self.n_gpus,
        )
        cuda = [str(x) for x in cuda]
        cuda = ",".join(cuda)
        self.cuda = cuda
        logger.info(f"获取到足够的卡，当前分配的卡为：{cuda}")

        # ---------------------------------------------------------------------------- #
        #                         从队列中弹出并注册处理器和进程
        # ---------------------------------------------------------------------------- #
        redis_client.pop_wait_queue(self.id)
        redis_client.register_process(self.id, cuda, n_gpus=self.n_gpus, memo=self.memo)

        return cuda

    @logger.catch
    def close(self):
        # 释放资源
        self.redis_client.deregister_process(self.id)


def is_processing_units_ready(devices, n_gpus):
    if len(devices) < n_gpus:
        # 没有符合条件的处理器
        return False
    else:
        return True

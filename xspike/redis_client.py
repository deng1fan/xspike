import json
import os
import datetime
import time
from redis import Redis
from loguru import logger


class RedisClient:
    def __init__(self):
        self.client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            charset="UTF-8",
            encoding="UTF-8",
        )

    def get_self_occupied_gpus(self, only_gpus=True):
        """
        获取自己已经占用的Gpu序号
        """
        self_occupied_gpus = self.client.hgetall("running_processes")
        if only_gpus:
            all_gpus = []
            for task in self_occupied_gpus.values():
                gpus = [
                    int(device) for device in str(json.loads(task)["cuda"]).split(",")
                ]
                all_gpus.extend(gpus)
            return list(set(all_gpus))
        return [json.loads(g) for g in self_occupied_gpus.values()]

    def join_wait_queue(self, id, n_gpus, memo):
        """
        加入等待队列
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        content = {
            "n_gpus": n_gpus,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "id": id,
            "task_desc": memo,
        }
        wait_num = len(self.client.lrange("wait_queue", 0, -1))
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            logger.info(f"正在排队中！ 目前排第一位！")
        else:
            logger.info(f"正在排队中！ 前方还有 {wait_num} 个任务！")
        return wait_num

    def is_my_turn(self, id):
        """
        排队这么长时间，是否轮到我了？
        """
        curr_task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        return curr_task["id"] == id

    def update_queue(self, id):
        """
        更新等待队列
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["id"] != id:
            # 登记异常信息
            logger.warning("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        curr_time = datetime.datetime.now()
        update_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        task["update_time"] = update_time
        self.client.lset("wait_queue", 0, json.dumps(task))

    def pop_wait_queue(self, id):
        """
        弹出当前排位第一的训练任务
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["id"] != id:
            # 登记异常信息
            logger.warning("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        next_task = self.client.lpop("wait_queue")
        return next_task

    def register_process(self, id, cuda, n_gpus, memo):
        """
        将当前训练任务登记到进程信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")

        content = {
            "cuda": cuda,
            "units_count": n_gpus,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "id": id,
            "task_desc": memo,
        }
        self.client.hset("running_processes", id, json.dumps(content))
        logger.info("成功登记进程使用信息到Redis服务器！")
        return id

    def deregister_process(self, id):
        """
        删除当前训练任务的信息
        """
        task = self.client.hget("running_processes", id)
        if task:
            self.client.hdel("running_processes", id)
            logger.info("成功删除Redis服务器上的进程使用信息！")
        else:
            logger.warning(
                "无法找到当前训练任务在Redis服务器上的进程使用信息！或许可以考虑检查一下Redis的数据 🤔"
            )

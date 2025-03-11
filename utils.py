import os
from rich.console import Console
import subprocess
from rich.progress import track
import time
import re
import sys
import traceback
from dingtalkchatbot.chatbot import DingtalkChatbot
import psutil
import copy
import argparse
from loguru import logger
from datetime import timedelta
import argparse
import json
import importlib
import inspect
from datetime import datetime, timezone
import ollama
import uuid
import random
from xspike.visualizer import visualize_text_clusters

console = Console()


def gen_id(digit=8):
    return str(uuid.uuid4())[:digit]


def chat_flow(msg_flow: list, model: list="llama3", system_prompt: str=""):
    """与 Ollama 对话

    Args:
        msg_flow (list): list 形式如下：
            [
                "你可以帮助我生成数据吗？",
                "生成金融领域对话数据",
                "分析对话话题转换",
            ]
        model (str, optional): _description_. Defaults to "llama3".

    Returns:
        _type_: _description_
    """
    if system_prompt != "":
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    for i, msg in enumerate(msg_flow):
        print(f"\n第{i+1}轮对话：{msg}")
        messages.append(
                {
                    "role": "user", "content": msg,
                }
            )
        messages.append(
                {
                    "role": "assistant", 
                    "content": ollama.chat(
                                    model=model,
                                    messages=messages,
                                    options = {"temperature":0.61,}
                                    )["message"]["content"],
                }
            )
        print("\n系统：" + messages[-1]["content"] + "\n")
    return messages[-1]["content"], messages


def chat(msg, model: str="llama3", system_prompt: str=""):
    """与 Ollama 对话

    Args:
        msg (_type_): str 或者 list，list 形式如下：
            [
                {"user": "你好。"},
                {"assistant": "你好，请问有什么可以帮助您的？"},
                {"user": "我想询问天气。"},
            ]
        model (str, optional): _description_. Defaults to "llama3".

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if system_prompt != "":
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    if isinstance(msg, str):
        messages.append(
                {
                    "role": "user",
                    "content": msg,
                }
            )
        response = ollama.chat(
            model=model,
            messages=messages,
        )
    elif isinstance(msg, list):
        for text in msg:
            messages.append({
                        "role": list(text.keys())[0],
                        "content": list(text.values())[0],
                    })
        response = ollama.chat(
            model=model,
            messages=messages,
            options = {"temperature":0.61,}
        )
    else:
        raise TypeError("msg must be str or list")
    return response["message"]["content"]


def set_env_variables(env_variables: dict):
    for key, value in env_variables.items():
        os.environ[key] = value
        logger.info(f"设置环境变量 {key} = {value}")


def print_error_info(e: Exception):
    """打印错误信息

    Args:
        e (_type_): 异常事件
    """
    print("str(Exception):\t", str(Exception))
    print("str(e):\t\t", str(e))
    print("repr(e):\t", repr(e))
    # Get information about the exception that is currently being handled
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("e.message:\t", exc_value)
    print(
        "Note, object e and exc of Class %s is %s the same."
        % (type(exc_value), ("not", "")[exc_value is e])
    )
    print("traceback.print_exc(): ", traceback.print_exc())
    print("traceback.format_exc():\n%s" % traceback.format_exc())


def load_module(module_path, class_name=None, method_name=None, return_instance=True, *args, **kwargs):
    """
    动态加载模块中的类或方法。

    :param module_path: 模块的完整路径，如 'mypackage.mymodule'。
    :param class_name: 可选，要加载的类的名称。
    :param method_name: 可选，要加载的方法的名称。
    :param return_instance: 是否返回类的实例。
    :return: 返回类实例或方法，取决于class_name和method_name的设置。
    """
    try:
        # 导入模块
        module = importlib.import_module(module_path)

        if class_name:
            # 加载类并创建实例
            if inspect.isclass(getattr(module, class_name, None)):
                if return_instance:
                    class_instance = getattr(module, class_name)(*args, **kwargs)
                    return class_instance
                else:
                    return getattr(module, class_name)
            else:
                raise AttributeError(f"{class_name} is not a class in {module_path}")
        elif method_name:
            # 加载方法
            if hasattr(module, method_name):
                method = getattr(module, method_name)
                return method
            else:
                raise AttributeError(
                    f"{method_name} is not a method or does not exist in {module_path}"
                )
        else:
            raise ValueError("Either class_name or method_name must be provided.")
    except ModuleNotFoundError:
        raise ImportError(f"Module {module_path} not found.")
    except Exception as e:
        raise e


def delete_between_delimiters(text, delimiter1, delimiter2, remove_delimiter=True):
    """删除text中位于delimiter1和delimiter2之间的字符

    Args:
        text (str): 文本
        delimiter1 (str): 前分隔符
        delimiter2 (str): 后分隔符
        remove_delimiter (bool, optional): 是否删除分隔符. Defaults to True.

    Returns:
        result: 处理后的文本
    """
    if remove_delimiter:
        pattern = re.escape(delimiter1) + r"(.*?)" + re.escape(delimiter2)
        result = re.sub(pattern, "", text)
        return result
    else:
        # 定义一个替换函数，用于保留分隔符
        def replace_func(match):
            # match.group(0) 是整个匹配的文本，包括分隔符
            return match.group(0).replace(match.group(1), "")  # 删除匹配的内层内容

        # 构建正则表达式模式
        pattern = re.escape(delimiter1) + r"(.*?)" + re.escape(delimiter2)
        # 使用正则表达式替换匹配的子串，保留分隔符
        return re.sub(pattern, replace_func, text)


def get_between_delimiters(text, delimiter1, delimiter2):
    # 构建正则表达式模式
    pattern = re.escape(delimiter1) + r"(.*?)" + re.escape(delimiter2)
    # 使用正则表达式查找所有匹配的子串
    matches = re.findall(pattern, text)
    # 返回所有匹配的子串列表
    return matches


def echo(msg, color="green"):
    console.print(msg, style=color)


def run_cmd_inactivate(cmd_list):
    if isinstance(cmd_list, str):
        cmd = cmd_list
        print("\n" + cmd)
        while True:
            exitcode = os.system(cmd)
            if exitcode != 0:
                echo("执行 {} 失败！".format(cmd), "#FF6AB3")
                echo("可通过在下方修改命令继续执行，或者直接按下回车键结束操作：")
                cmd = input()
                if cmd == "":
                    return exitcode
            else:
                return exitcode

    outputs = []
    for cmd in track(cmd_list, description="命令执行中", transient=True):
        print("\n" + cmd)
        while True:
            exitcode = os.system(cmd)
            if exitcode != 0:
                echo("执行 {} 失败！".format(cmd), "#FF6AB3")
                echo("可通过在下方修改命令继续执行，或者直接按下回车键结束操作：")
                cmd = input()
                if cmd == "":
                    break
            else:
                break

    return outputs


def run_cmd(cmd_list, show_cmd=True):
    if isinstance(cmd_list, str):
        cmd = cmd_list
        if show_cmd:
            print("\n" + cmd)
        while True:
            exitcode, output = subprocess.getstatusoutput(cmd)
            if exitcode != 0:
                echo("执行 {} 失败！".format(cmd), "#FF6AB3")
                echo("错误信息：\n{}".format(output))
                echo("可通过在下方修改命令继续执行，或者直接按下回车键结束操作：")
                cmd = input()
                if cmd == "":
                    return output
            else:
                return output

    outputs = []
    for cmd in track(cmd_list, description="命令执行中", transient=True):
        if show_cmd:
            print("\n" + cmd)
        while True:
            exitcode, output = subprocess.getstatusoutput(cmd)
            if exitcode != 0:
                echo("执行 {} 失败！".format(cmd), "#FF6AB3")
                echo("错误信息：\n{}".format(output))
                echo("可通过在下方修改命令继续执行，或者直接按下回车键结束操作：")
                cmd = input()
                if cmd == "":
                    outputs.append(output)
                    break
            else:
                outputs.append(output)
                break

    return outputs


def hi():
    """Prints a banner."""
    # Banner
    print("")
    print("")
    print("")
    print("")
    print("")
    print("         __                         __  ___      __                ____")
    print(
        "        / /   ____ _____  __  __   /  |/  /___ _/ /_____  _____   / __ )__  _________  __"
    )
    print(
        "       / /   / __ `/_  / / / / /  / /|_/ / __ `/ //_/ _ \\/ ___/  / __  / / / / ___/ / / /"
    )
    print(
        "      / /___/ /_/ / / /_/ /_/ /  / /  / / /_/ / ,< /  __(__  )  / /_/ / /_/ (__  ) /_/ /"
    )
    print(
        "     /_____/\\__,_/ /___/\\__, /  /_/  /_/\\__,_/_/|_|\\___/____/  /_____/\\__,_/____/\\__, /"
    )
    print(
        "                       /____/                                                   /____/"
    )
    print("")
    print("")
    print("")
    print("")


def notice(msg: str = "", warning=False, access_token="", secret=""):
    """钉钉消息通知"""
    access_token = (
        os.environ.get("DINGDING_ACCESS_TOKEN", "")
        if access_token == ""
        else access_token
    )
    secret = os.environ.get("DINGDING_SECRET", "") if secret == "" else secret
    if access_token == "" or secret == "":
        logger.warning("未设置钉钉Token，无法发送消息: " + msg)
        logger.warning("请在环境变量中设置 DINGDING_ACCESS_TOKEN 和 DINGDING_SECRET !")
        logger.warning("例如：export DINGDING_ACCESS_TOKEN=your_access_token")
        logger.warning("或者在调用函数时传入 access_token 和 secret 参数！")
        return

    pid = os.getpid()
    proctitle = psutil.Process(pid).name()
    if warning:
        msg = f"⚠️\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"
    else:
        msg = f"🪼\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"

    # WebHook地址
    webhook = f"https://oapi.dingtalk.com/robot/send?access_token={access_token}"
    xiaoding = DingtalkChatbot(webhook, secret=secret, pc_slide=True)
    # Text消息@所有人
    xiaoding.send_text(msg=msg)
    logger.info(f"已将下面通知发送到钉钉！")
    logger.info(msg)


class Result(dict):
    """返回结果基类

    Args:
        dict (_type_): 初始化字典
    """

    def __getattr__(self, name):
        try:
            # 尝试返回字典中对应的值
            return self[name]
        except KeyError:
            # 如果键不存在，抛出AttributeError
            raise AttributeError(f"No attribute '{name}'")

    def __init__(self, *args, **kwargs):
        super(Result, self).__init__()
        for arg in args:
            for key, value in arg.items():
                self[key] = value
        self.add(**kwargs)

    # 序列化时调用
    def __getstate__(self):
        return None

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def delete(self, keys):
        for k in keys:
            self.pop(k)

    def merge(self, merge_dict):
        if not isinstance(merge_dict, Result) and not isinstance(merge_dict, dict):
            raise TypeError("不支持的合并类型")
        for k, v in merge_dict.items():
            if k in ["msg", "status"] or k in self:
                continue
            self[k] = v

    def merge_or_update(self, merge_dict):
        if not isinstance(merge_dict, Result) and not isinstance(merge_dict, dict):
            raise TypeError("不支持的合并类型")
        for k, v in merge_dict.items():
            if k in ["msg", "status"]:
                continue
            self[k] = v

    @staticmethod
    def create_error_msg_result(msg="Error Result", **kwargs):
        result = Result()
        result["msg"] = msg
        result["status"] = False
        result.add(**kwargs)
        return result

    def get(self, name, other=None):
        if name is None:
            return list(self.values())
        elif isinstance(name, str):
            return self[name] if name in self else other
        elif isinstance(name, list):
            values = [self[n] for n in name]
            return values
        else:
            return self.create_error_msg_result(msg=f"Key值类型{type(name)}不支持")

    def print(self, name=None):
        print("  =====" + self["msg"] + "=====")
        values = self.get(name)
        if name is None:
            name = list(self.keys())
        for i, k in enumerate(name):
            v = values[i]
            print(f"  {k}:    {v}")
        print("  =====" + self["msg"] + "=====")

    def flatten_to_print(self):
        value_str = ""
        keys = self.keys()
        for i, k in enumerate(keys):
            v = self[k]
            value_str = value_str + k + " : " + str(v) + "\n"
        return value_str

    def aprintend_values(self, next_dict):
        if not isinstance(next_dict, Result) and not isinstance(next_dict, dict):
            raise TypeError("不支持的合并类型")
        for key in next_dict.keys():
            if key not in self.keys():
                self[key] = []

            (
                self[key].aprintend(next_dict[key])
                if isinstance(self[key], list)
                else [self[key]].aprintend(next_dict[key])
            )

    def str(self, key_name, default_value=""):
        return self.get(key_name, default_value)

    def bool(self, key_name, default_value=False):
        return self.get(key_name, default_value)

    def int(self, key_name, default_value=0):
        return self.get(key_name, default_value)

    def float(self, key_name, default_value=0.0):
        return self.get(key_name, default_value)

    def list(self, key_name, default_value=[]):
        return self.get(key_name, default_value)

    def dict(self, key_name, default_value={}):
        return self.get(key_name, default_value)

    def get_dict(self):
        dict_item = {}
        for key, value in self.items():
            dict_item[key] = value
        return dict_item

    def set(self, key_name, value):
        self[key_name] = value

    def set_with_dict(self, dict_value):
        for key, value in dict_value.items():
            if "." in key:
                key_list = key.split(".")
                self[key_list[0]][key_list[1]] = value
            else:
                self[key] = value

    def __deepcopy__(self, memo=None, _nil=[]):
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        dict = Result()
        memo[d] = id(dict)
        for key in self.keys():
            dict.__setattr__(
                copy.deepcopy(key, memo), copy.deepcopy(self.__getattr__(key), memo)
            )
        return dict

    def copy(self):
        return super().copy()


def get_file_paths_in_directory(directory="./", ignored_files=[], only_files=[]):
    file_paths = []
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件的扩展名
            _, file_extension = os.path.splitext(file)
            if len(only_files) > 0 and file_extension.lower() in only_files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                continue
            # 如果文件的扩展名不在排除列表中，则添加到列表中
            if len(only_files) == 0 and file_extension.lower() not in ignored_files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths


def get_args_value(argument_key):
    args = parse_command_line_args()
    if argument_key in args:
        value = args[argument_key]
    else:
        value = None

    return value


def generate_timestamp():
    # 获取当前时间，并确保是UTC时间，避免时区问题
    now = datetime.now(timezone.utc)
    # 格式化时间戳，去掉空格和不安全的字符
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_str


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        # 开始计时，记录开始时间
        self.start_time = time.time()

    def end(self):
        # 结束计时，记录结束时间
        if self.start_time is None:
            raise ValueError("计时器尚未开始，请先调用start方法。")
        self.end_time = time.time()
        # 计算时间间隔
        elapsed_seconds = self.end_time - self.start_time
        # 将秒数转换为天数、小时、分钟和秒数
        timedelta_obj = timedelta(seconds=elapsed_seconds)
        days = timedelta_obj.days
        # 初始化remainder为0，以避免在divmod返回值为0时出现问题
        remainder = timedelta_obj.seconds
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        # 返回格式化的时间间隔
        return days, hours, minutes, seconds


def parse_command_line_args():
    """
    解析命令行参数，并将参数组装成字典。支持解析值为字典、列表、整数和浮点数的参数。
    """
    args_dict = {}
    for i in range(1, len(sys.argv), 2):
        if i + 1 < len(sys.argv) and sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            value = sys.argv[i + 1]
            # 判断参数的值是否为字典格式
            if value.startswith("{") and value.endswith("}"):
                # 解析为字典
                value_dict = eval(value)
                args_dict[key] = value_dict
            # 判断参数的值是否为列表格式
            elif value.startswith("[") and value.endswith("]"):
                # 解析为列表
                value_list = eval(value)
                args_dict[key] = value_list
            # 判断参数的值是否为整数格式
            elif value.isdigit():
                # 解析为整数
                value_int = int(value)
                args_dict[key] = value_int
            # 判断参数的值是否为浮点数格式
            elif "." in value and value.replace(".", "", 1).isdigit():
                # 解析为浮点数
                value_float = float(value)
                args_dict[key] = value_float
            else:
                args_dict[key] = value
    # dict 中每个元素转为成员变量
    for key, value in args_dict.items():
        if isinstance(value, dict):
            args_dict[key] = Result(value)
            
        elif isinstance(value, list):
            args_dict[key] = [Result(item) if isinstance(item, dict) else item for item in value]
            
    args_dict = Result(args_dict)
    return args_dict



class BaseArgs:
    def __init__(self):
        self.args = parse_command_line_args()
        self.params = {}
        
        
    def args_update(self):
        """
        参数填充。
        """
        for k, v in self.params.items():
            if k in self.args:
                if isinstance(self.args[k], int) or isinstance(self.args[k], float):
                    self.params[k] = str(self.args[k])
                else:
                    self.params[k] = str(self.args[k])
        return self.params
        

    def parse_args(self):
        """
        解析命令行参数。
        """
        args = parse_command_line_args()
        self.update_or_add(args)

    def values(self):
        """
        返回参数字典。
        """
        return self.params

    def has(self, key):
        """
        判断参数字典中是否存在指定键。


        :param key: 要判断的键。
        :return: 如果存在，返回 True；否则，返回 False。
        """
        return key in self.params

    def check(self, key):
        """
        检查参数字典中是否存在指定键，并且值不为空。


        :param key: 要检查的键。
        :return: 如果存在并且不为空，返回 True；否则，返回 False。
        """
        if (
            key in self.params
            and self.params[key] is not None
            and self.params[key] != ""
        ):
            return True
        else:
            return False

    def set(self, key, value=None):
        """
        设置参数值。
        :param key: 参数名称。
        :param value: 参数值。
        """
        self.params[key] = value
        logger.info(f"Set parameter '{key}' with value '{value}'.")

    def get(self, key, default_value=None):
        return self.params.get(key, default_value)

    def update(self, args):
        """
        动态更新参数值。

        :param args: 包含命令行参数的对象。
        """
        for attr, value in args.items():
            if attr in self.params:
                self.params[attr] = value
                logger.info(f"Updated parameter '{attr}' with value '{value}'.")

    def update_or_add(self, args):
        """
        动态更新或添加参数值。

        :param args: 包含命令行参数的对象。
        """
        for attr, value in args.items():
            if attr in self.params:
                self.params[attr] = value
                logger.info(f"Updated parameter '{attr}' with value '{value}'.")
            else:
                self.params[attr] = value
                logger.info(f"Added parameter '{attr}' with value '{value}'.")

    def delete_parameter(self, parameter_name):
        """
        删除一个参数。

        :param parameter_name: 要删除的参数名称。
        """
        if parameter_name in self.params:
            del self.params[parameter_name]
            logger.info(f"Deleted parameter '{parameter_name}'.")
        else:
            logger.warning(f"Parameter '{parameter_name}' does not exist.")

    def print(self):
        """
        使用 json.dump 美化打印所有参数及其值。
        """
        print(json.dumps(self.params, indent=4, sort_keys=True))

    def __str__(self):
        """
        返回参数的字符串表示。
        """
        return json.dumps(self.params, indent=4)


def inspect_args(class_or_func):
    return inspect.signature(class_or_func).parameters.keys()


def inspect_and_filter_args(class_or_func, args):
    """
    过滤掉不在函数参数列表中的参数。


    :param class_or_func: 类或函数。
    :param args: 包含命令行参数的字典。
    :return: 过滤后的参数字典。
    """
    arg_names = inspect_args(class_or_func)
    filtered_args = {}
    for arg_name in arg_names:
        if arg_name in args:
            filtered_args[arg_name] = args[arg_name]
    return filtered_args


def list_to_dict(lst):
    # keys = data_list[0].keys()
    # res = dict()
    # for f in data_list:
    #     for key in keys:
    #         if key not in res:
    #             res[key] = []
    #         item = [f[key]] if isinstance(f[key], int) else f[key]
    #         res[key].append(item)
    # return res
    if not lst:
        return {}

    # 获取所有字典的共有 key
    common_keys = set(lst[0].keys())
    for d in lst[1:]:
        common_keys.intersection_update(d.keys())

    # 构建新的字典
    common_dict = {key: [] for key in common_keys}
    for d in lst:
        for key in common_keys:
            common_dict[key].append(d[key])

    return common_dict

def dict_to_list(data_dict):
    keys = list(data_dict.keys())
    res = []
    for i in range(len(data_dict[keys[0]])):
        item = dict()
        for key in keys:
            item[key] = data_dict[key][i]
        res.append(item)
    return res


def count_chars(text):
    """统计文本中的汉字和非汉字字符数"""
    chinese_chars = re.findall('[\u4e00-\u9fff]', text)
    non_chinese_chars = re.findall('[^\u4e00-\u9fff]', text)
    
    return len(chinese_chars), len(non_chinese_chars)


def edit_distance(s1, s2):
    # 创建一个 (len(s1)+1) x (len(s2)+1) 的二维数组
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # 初始化边界值
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # 动态规划计算编辑距离
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    return dp[len(s1)][len(s2)]


def average_edit_distance(reference_text, text_list):
    if not text_list:
        return 0.0

    total_distance = 0
    for text in text_list:
        total_distance += edit_distance(reference_text, text)

    return total_distance / len(text_list)



def centralize_data(data, centralize_text, center_num=10, min_member_size=100, max_member_size=2000):
    """
    为每个样本添加中心样本
    Parameters:
        data: list, 样本列表，每个元素都是一个字典
    """
    # ---------------------------------------------------------------------------- #
    #                         聚类
    # ---------------------------------------------------------------------------- #
    for i, d in enumerate(data):
        d["sn_code"] = gen_id()
    format_data = list_to_dict(data)
    cluster = visualize_text_clusters(format_data[centralize_text])
    center_data_index = [i["center_index"] for i in cluster]

    def get_center_id(id):
        for i in cluster:
            if id in i["cluster_indices"]:
                return i["center_index"]

    # ---------------------------------------------------------------------------- #
    #                         添加中心样本
    # ---------------------------------------------------------------------------- #
    all_center_ids = {}
    for i, d in enumerate(data):
        all_center_ids[i] = 0
    for i, d in enumerate(data):
        if i in center_data_index:
            continue
        center_id = get_center_id(i)
        if center_id in center_data_index:
            all_center_ids[center_id] += 1
    center_ids = []
    for k, v in all_center_ids.items():
        if v > min_member_size:
            center_ids.append(k)

    center_ids = center_ids[:center_num]
    print("中心样本数量：", len(center_ids))
    print("每个中心样本的成员数量：", [all_center_ids[k] for k in center_ids])

    id_count = {}
    new_data = []
    for i in center_ids:
        id_count[i] = 0
    for i, d in enumerate(data):
        c_id = get_center_id(i)
        if (
            i in center_ids
            or c_id not in center_ids
            or id_count[c_id] >= max_member_size
        ):
            continue
        new_data.append(d)
        new_data[-1]["center_data"] = data[c_id]
        id_count[c_id] += 1

    print("数据总数：", len(new_data))
    return new_data



import os
from rich.console import Console
import subprocess
from rich.progress import track
import time
import logging
import inspect
import re
from importlib import import_module
import sys
import traceback
from dingtalkchatbot.chatbot import DingtalkChatbot
import psutil
import copy
import argparse


console = Console()

class Logger:
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[91m',
        'FILENAME': '\033[95m',
        'LINENO': '\033[95m',
    }
    RESET = '\033[0m'

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.rank = 0

    def get_logger(self):
        return self.logger

    def set_rank(self, rank):
        self.rank = rank

    def format_with_color(self, message, color):
        return f'{color}{message}{self.RESET}'

    def debug(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['DEBUG'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["DEBUG"]}[DEBUG] {formatted_time}  {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def info(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['INFO'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[2m{formatted_time} {self.COLORS["INFO"]}{self.RESET} \033[2m[{formatted_filename}:\033[2m{formatted_lineno}] {message}')

    def warning(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['WARNING'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["WARNING"]}[WARNING] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def error(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['ERROR'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["ERROR"]}[ERROR] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def critical(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['CRITICAL'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["CRITICAL"]}[CRITICAL] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')



log = Logger(__name__)


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



def load_class(class_path: str) -> type:
    """Load a class from a string.

    Args:
        class_path: The class path.

    Returns:
        The class.

    Raises:
        ImportError: If the class cannot be imported.
        AttributeError: If the class cannot be found.
    """
    try:
        module_path, _, class_name = class_path.rpartition(".")
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        log.error(f"加载类失败: {class_path}")
        print_error_info(e)
        raise e





def delete_between_delimiters(text, delimiter1, delimiter2):
    """删除text中位于delimiter1和delimiter2之间的字符

    Args:
        text (str): 文本
        delimiter1 (str): 前分隔符
        delimiter2 (str): 后分隔符

    Returns:
        result: 处理后的文本
    """
    pattern = re.escape(delimiter1) + ".*?" + re.escape(delimiter2)
    result = re.sub(pattern, '', text)
    return result


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
    """Prints a banner.
    """
    # Banner
    print("")
    print("")
    print("")
    print("")
    print("")
    print("         __                         __  ___      __                ____")
    print("        / /   ____ _____  __  __   /  |/  /___ _/ /_____  _____   / __ )__  _________  __")
    print("       / /   / __ `/_  / / / / /  / /|_/ / __ `/ //_/ _ \\/ ___/  / __  / / / / ___/ / / /")
    print("      / /___/ /_/ / / /_/ /_/ /  / /  / / /_/ / ,< /  __(__  )  / /_/ / /_/ (__  ) /_/ /")
    print("     /_____/\\__,_/ /___/\\__, /  /_/  /_/\\__,_/_/|_|\\___/____/  /_____/\\__,_/____/\\__, /")
    print("                       /____/                                                   /____/")
    print("")
    print("")
    print("")
    print("")




def notice(msg: str = "", warning=False, access_token="", secret=""):
    """钉钉消息通知
    
    """
    access_token = os.environ.get('DINGDING_ACCESS_TOKEN', "") if access_token == "" else access_token
    secret = os.environ.get('DINGDING_SECRET', "") if secret == "" else secret
    if access_token == "" or secret == "":
        log.warning("未设置钉钉Token，无法发送消息: " + msg)
        log.warning("请在环境变量中设置 DINGDING_ACCESS_TOKEN 和 DINGDING_SECRET !")
        log.warning("例如：export DINGDING_ACCESS_TOKEN=your_access_token")
        log.warning("或者在调用函数时传入 access_token 和 secret 参数！")
        return
    
    pid = os.getpid()
    proctitle = psutil.Process(pid).name()
    if warning:
        msg = f"⚠️\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"
    else:
        msg = f"🪼\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"
    
    # WebHook地址
    webhook = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'
    xiaoding = DingtalkChatbot(webhook, secret=secret, pc_slide=True)
    # Text消息@所有人
    xiaoding.send_text(msg=msg)
    log.info(f"已将下面通知发送到钉钉！")
    log.info(msg)
    

class Result(dict):
    """返回结果基类

    Args:
        dict (_type_): 初始化字典
    """
    def __getattr__(self, name):
        return self[name]

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

            self[key].aprintend(next_dict[key]) if isinstance(self[key], list) else [
                self[key]
            ].aprintend(next_dict[key])

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
                copy.deepcopy(key, memo), copy.deepcopy(
                    self.__getattr__(key), memo)
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


def get_argument_value(argument_key):
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Get a specific argument value from command line.")
    
    # 添加参数，这里假设我们想要获取的键是'--flag'
    parser.add_argument(f'-{argument_key}', help=f'The value for the {argument_key} argument.')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取指定的参数值
    value = getattr(args, argument_key, None)
    
    return value

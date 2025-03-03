import os
from xspike.utils import run_cmd, echo
import datetime
import time


def run():

    project_path = os.getcwd()
    echo("")
    echo("")
    echo("")
    echo("")
    echo("")
    echo("         __                         __  ___      __                ____")
    echo(
        "        / /   ____ _____  __  __   /  |/  /___ _/ /_____  _____   / __ )__  _________  __"
    )
    echo(
        "       / /   / __ `/_  / / / / /  / /|_/ / __ `/ //_/ _ \\/ ___/  / __  / / / / ___/ / / /"
    )
    echo(
        "      / /___/ /_/ / / /_/ /_/ /  / /  / / /_/ / ,< /  __(__  )  / /_/ / /_/ (__  ) /_/ /"
    )
    echo(
        "     /_____/\\__,_/ /___/\\__, /  /_/  /_/\\__,_/_/|_|\\___/____/  /_____/\\__,_/____/\\__, /"
    )
    echo(
        "                       /____/                                                   /____/"
    )
    echo("")
    echo("")
    echo("")
    echo("")
    echo("当前目录下（{}）共发现以下实验计划：".format(project_path))

    # 获取可选的项目名
    projects = os.listdir(os.path.join(project_path, "exp_plans/"))
    for i, project in enumerate(projects):
        echo("{}、 {}".format(i + 1, project.split(".")[0]), "#FF6AB3")

    # ---------------------------------------------------------------------------- #
    #                         获取实验计划
    # ---------------------------------------------------------------------------- #
    echo("\n请选择要启动的实验计划：")
    exp_num = int(input())
    exp_name = projects[exp_num - 1].split(".")[0]
    exp_plan_path = project_path + "/exp_plans/" + projects[exp_num - 1]
    log_path = os.path.join(project_path, "logs/{}".format(exp_name))

    # 读取计划命令列表
    ori_exp_plan = open(exp_plan_path, "r").readlines()
    cmd = ""
    exp_plan = []
    for idx, line in enumerate(ori_exp_plan):
        if line == "\n":
            if cmd != "":
                exp_plan.append(cmd)
            cmd = ""
        else:
            if line.startswith("#"):
                continue
            cmd += line.strip().replace("\n", "").replace("\\", " ")
    if cmd != "":
        exp_plan.append(cmd)
    exp_plan_num = len(exp_plan)
    echo("\n计划共启动 {} 个实验！\n".format(exp_plan_num))

    for i, line in enumerate(exp_plan):
        exp_log_path = os.path.join(
            project_path,
            "{}/{}.log".format(
                log_path, "{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            ),
        )

        parent_path = "/".join(exp_log_path.split("/")[:-1])
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        echo("实验 {} ：".format(str(i + 1)), "#2EA1AC")
        cmd = "{} > {} 2>&1 &".format(line, exp_log_path)
        run_cmd(cmd, show_cmd=True)
        echo("\n查看实验 {} 的日志：tail -f {}".format(str(i + 1), exp_log_path))
        echo("\n")
        if i + 1 != exp_plan_num:
            time.sleep(10)
            echo(" " * 5 + "-" * 70 + "\n", "#CB4B15")

    echo("\n所有实验均已启动！不如趁现在去喝杯咖啡！", "green")

import comet_ml
import os
from xspike.utils import get_file_paths_in_directory
import uuid

from loguru import logger


ONLY_FILES = [".py", ".yml"]


class CometClient:

    def __init__(
        self, project_name="Default", workspace=None, api_key=None, exp_name=None
    ):
        self.project_name = project_name
        self.workspace = workspace
        self.api_key = api_key
        # 生成实验独一无二的 ID
        self.experiment_key = str(uuid.uuid4()).replace("-", "")
        os.environ["COMET_EXPERIMENT_KEY"] = self.experiment_key

        self.experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=project_name,
            experiment_key=self.experiment_key,
            display_summary_level=0,
            log_git_patch=False,
            log_git_metadata=False,
        )
        self.experiment.set_name(exp_name)
        self.experiment.log_other("进程ID", str(os.getpid()))

        logger.info("Comet 实验记录已启动")

    def get_experiment_by_key(self):
        api = comet_ml.api.API()
        experiment = api.get_experiment_by_key(os.environ["COMET_EXPERIMENT_KEY"])
        return experiment

    def get_experiment(self):
        return self.experiment

    def log_directory(self, directory, only_files=ONLY_FILES):
        file_paths = get_file_paths_in_directory(directory, only_files=only_files)
        logger.info(f"目录 {directory} 下的文件正准备上传至 Comet，这可能需要一段时间")
        for file_path in file_paths:
            try:
                self.experiment.log_asset(file_path, file_name=file_path)
            except Exception as e:
                logger.warning(f"文件 {file_path} 上传至 Comet 失败: {e}")
                logger.warning(f"请检查文件是否过大，或是否有特殊字符")

    def log_dict(self, dictionary, prefix=""):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.log_dict(value, prefix=f"{prefix}{key}/")
            else:
                self.experiment.log_parameter(f"{prefix}{key}", value)

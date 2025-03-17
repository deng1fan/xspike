import os
import pickle
import json
import jsonlines
import pandas as pd
from loguru import logger
import importlib
from typing import Any, Dict, List, Optional, Union


def check_module(module_name: str) -> bool:
    """检查指定模块是否可用

    Args:
        module_name: 模块名称

    Returns:
        bool: 模块是否可用
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


@logger.catch
def load_in(path: str, data_name: str = "", encoding: str = "utf-8", **kwargs) -> Any:
    """读取文件，根据文件后缀名自动选择读取方法
    支持的格式: 'pkl'、'txt'、'json', 'jsonl'、'csv', 'xlsx', 'npy', 'npz'等

    Args:
        path: 文件路径
        data_name: 打印提示时需要，便于控制台查看加载的是什么文件, 默认为空
        encoding: 文本文件的编码方式，默认为utf-8
        **kwargs: 传递给特定加载函数的额外参数

    Returns:
        加载的数据对象

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
        ImportError: 缺少必要的依赖库
    """
    if not os.path.exists(path):
        logger.error(f"文件路径不存在: {path}")
        raise FileNotFoundError(f"文件不存在: {path}")

    if not data_name:
        data_name = os.path.basename(path)

    logger.info(f"正在加载文件 {data_name} from {path}")

    # 获取文件扩展名
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    # 根据文件扩展名选择相应的加载方法
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    
    elif ext == ".csv":
        data = pd.read_csv(path, **kwargs)
    
    elif ext in (".xlsx", ".xls"):
        data = pd.read_excel(path, **kwargs)
    
    elif ext == ".json":
        try: 
            # 尝试按标准JSON读取
            with open(path, "r", encoding=encoding) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # 如果标准读取失败，尝试按行读取JSON
            data = []
            with open(path, "r", encoding=encoding) as file:
                for line in file:
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"忽略无效的JSON行: {line[:30]}...")
    
    elif ext == ".jsonl":
        data = []
        with jsonlines.open(path, mode="r") as reader:
            for item in reader:
                data.append(item)
    
    elif ext == ".txt":
        data = []
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(line)
        
        # 如果只有一行，且没有要求保持列表格式，返回字符串而不是列表
        if kwargs.get("as_string", False) or (len(data) == 1 and kwargs.get("single_as_string", True)):
            data = "\n".join(data)
    
    elif ext in (".npy", ".npz"):
        if check_module("numpy"):
            import numpy as np
            if ext == ".npy":
                data = np.load(path, **kwargs)
            else:  # .npz
                npz_file = np.load(path, **kwargs)
                if data_name and data_name != os.path.basename(path):
                    # 如果提供了特定数据名称，则只加载该数组
                    data = npz_file[data_name]
                else:
                    data = npz_file
        else:
            logger.error("加载NPY/NPZ文件需要安装numpy库")
            raise ImportError("加载NPY/NPZ文件需要安装numpy库")
    
    elif ext in (".h5", ".hdf5"):
        if check_module("h5py"):
            import h5py
            h5_file = h5py.File(path, 'r')
            if data_name and data_name != os.path.basename(path):
                data = h5_file[data_name][()]
                h5_file.close()
            else:
                data = h5_file  # 注意：调用者需要负责关闭文件
        else:
            logger.error("加载HDF5文件需要安装h5py库")
            raise ImportError("加载HDF5文件需要安装h5py库")
    
    elif ext == ".parquet":
        if check_module("pyarrow") or check_module("fastparquet"):
            data = pd.read_parquet(path, **kwargs)
        else:
            logger.error("加载Parquet文件需要安装pyarrow或fastparquet库")
            raise ImportError("加载Parquet文件需要安装pyarrow或fastparquet库")
    
    elif ext == ".feather":
        if check_module("pyarrow"):
            data = pd.read_feather(path, **kwargs)
        else:
            logger.error("加载Feather文件需要安装pyarrow库")
            raise ImportError("加载Feather文件需要安装pyarrow库")
    
    else:
        logger.error(f"不支持的文件格式: {ext}")
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"成功加载 {data_name}!")
    return data


@logger.catch
def save_as(
    data: Any, 
    save_path: str, 
    data_name: str = "", 
    encoding: str = "utf-8",
    create_dir: bool = True,
    protocol: int = 4, 
    **kwargs
) -> str:
    """将参数中的文件对象保存为指定格式文件
    支持的格式: 'pkl'、'txt'、'json', 'jsonl', 'csv', 'xlsx', 'npy', 'npz'等

    Args:
        data: 要保存的文件对象
        save_path: 文件的保存路径，应当包含文件名
        data_name: 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空
        encoding: 文本文件的编码方式，默认为utf-8
        create_dir: 是否自动创建父目录，默认为True
        protocol: pickle序列化协议版本，默认为4
        **kwargs: 传递给特定保存函数的额外参数

    Returns:
        str: 保存成功的文件路径

    Raises:
        ValueError: 不支持的文件格式
        ImportError: 缺少必要的依赖库
        OSError: 文件路径创建失败
    """
    if not data_name:
        data_name = os.path.basename(save_path)
    
    # 获取文件扩展名
    _, ext = os.path.splitext(save_path)
    ext = ext.lower()
    
    # 创建父目录（如果不存在且需要创建）
    parent_path = os.path.dirname(save_path)
    if parent_path and not os.path.exists(parent_path):
        if create_dir:
            logger.info(f"保存路径的父文件夹（{parent_path}）不存在，将自动创建....")
            try:
                os.makedirs(parent_path)
            except OSError as e:
                logger.error(f"创建目录失败: {e}")
                raise
        else:
            logger.error(f"保存路径的父文件夹（{parent_path}）不存在，且create_dir=False")
            raise FileNotFoundError(f"目录不存在: {parent_path}")
    
    logger.info(f"正在保存文件 {data_name} 到 {save_path}")
    
    if ext in (".pkl", ".pickle"):
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    
    elif ext == ".txt":
        with open(save_path, "w", encoding=encoding) as f:
            if isinstance(data, (list, tuple)):
                for line in data:
                    f.write(f"{line}\n")
            else:
                f.write(f"{data}\n")
    
    elif ext == ".json":
        with open(save_path, "w", encoding=encoding) as f:
            json.dump(
                data, 
                f, 
                ensure_ascii=kwargs.get("ensure_ascii", False),
                indent=kwargs.get("indent", 4),
                cls=kwargs.get("cls", None)
            )
    
    elif ext == ".jsonl":
        with jsonlines.open(save_path, mode="w") as writer:
            # 如果数据是列表或可迭代对象，使用write_all
            if isinstance(data, (list, tuple)) or (hasattr(data, '__iter__') and not isinstance(data, dict)):
                writer.write_all(data)
            else:
                # 否则写入单个对象
                writer.write(data)
    
    elif ext == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(save_path, index=kwargs.get("index", False), **{k:v for k,v in kwargs.items() if k != "index"})
        elif isinstance(data, (list, tuple)):
            # 如果是字典列表或其他可转换为DataFrame的对象，先转换
            try:
                pd.DataFrame(data).to_csv(save_path, index=kwargs.get("index", False))
            except (ValueError, TypeError):
                # 如果无法转换为DataFrame，则直接写入CSV
                import csv
                with open(save_path, 'w', newline='', encoding=encoding) as f:
                    writer = csv.writer(f)
                    if all(isinstance(row, (list, tuple)) for row in data):
                        writer.writerows(data)
                    else:
                        # 每个元素作为单独的行
                        for item in data:
                            writer.writerow([item])
        else:
            # 尝试转换其他类型的数据
            try:
                pd.DataFrame(data).to_csv(save_path, index=kwargs.get("index", False))
            except (ValueError, TypeError):
                logger.warning(f"无法将{type(data).__name__}类型转换为DataFrame，尝试其他方式保存")
                import csv
                with open(save_path, 'w', newline='', encoding=encoding) as f:
                    writer = csv.writer(f)
                    writer.writerow([data])
    
    elif ext in (".xlsx", ".xls"):
        if isinstance(data, pd.DataFrame):
            data.to_excel(save_path, index=kwargs.get("index", False), **{k:v for k,v in kwargs.items() if k != "index"})
        elif isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
            # 如果是DataFrame字典，保存为多个表
            with pd.ExcelWriter(save_path) as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=str(sheet_name), index=kwargs.get("index", False))
        else:
            # 尝试转换为DataFrame
            try:
                pd.DataFrame(data).to_excel(save_path, index=kwargs.get("index", False))
            except (ValueError, TypeError):
                logger.error(f"无法将{type(data).__name__}类型转换为DataFrame保存为Excel")
                raise TypeError(f"无法将{type(data).__name__}类型转换为DataFrame保存为Excel")
    
    elif ext in (".npy", ".npz"):
        if check_module("numpy"):
            import numpy as np
            if ext == ".npy":
                np.save(save_path, data)
            else:  # .npz
                if data_name and data_name != os.path.basename(save_path):
                    # 如果提供了特定数据名称，则保存为命名数组
                    np.savez(save_path, **{data_name: data})
                elif isinstance(data, dict):
                    # 如果是字典，则每个键作为单独的数组
                    np.savez(save_path, **data)
                else:
                    # 否则使用默认名称
                    np.savez(save_path, data=data)
        else:
            logger.error("保存NPY/NPZ文件需要安装numpy库")
            raise ImportError("保存NPY/NPZ文件需要安装numpy库")
    
    elif ext in (".h5", ".hdf5"):
        if check_module("h5py"):
            import h5py
            with h5py.File(save_path, 'w') as h5_file:
                if data_name and data_name != os.path.basename(save_path):
                    h5_file.create_dataset(data_name, data=data)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        h5_file.create_dataset(str(key), data=value)
                else:
                    h5_file.create_dataset('data', data=data)
        else:
            logger.error("保存HDF5文件需要安装h5py库")
            raise ImportError("保存HDF5文件需要安装h5py库")
    
    elif ext == ".parquet":
        if check_module("pyarrow") or check_module("fastparquet"):
            if isinstance(data, pd.DataFrame):
                data.to_parquet(save_path, **kwargs)
            else:
                try:
                    pd.DataFrame(data).to_parquet(save_path, **kwargs)
                except (ValueError, TypeError):
                    logger.error(f"无法将{type(data).__name__}类型转换为DataFrame保存为Parquet")
                    raise TypeError(f"无法将{type(data).__name__}类型转换为DataFrame保存为Parquet")
        else:
            logger.error("保存Parquet文件需要安装pyarrow或fastparquet库")
            raise ImportError("保存Parquet文件需要安装pyarrow或fastparquet库")
    
    elif ext == ".feather":
        if check_module("pyarrow"):
            if isinstance(data, pd.DataFrame):
                data.to_feather(save_path, **kwargs)
            else:
                try:
                    pd.DataFrame(data).to_feather(save_path, **kwargs)
                except (ValueError, TypeError):
                    logger.error(f"无法将{type(data).__name__}类型转换为DataFrame保存为Feather")
                    raise TypeError(f"无法将{type(data).__name__}类型转换为DataFrame保存为Feather")
        else:
            logger.error("保存Feather文件需要安装pyarrow库")
            raise ImportError("保存Feather文件需要安装pyarrow库")
    
    else:
        logger.error(f"不支持的文件格式: {ext}")
        raise ValueError(f"不支持的文件格式: {ext}")
    
    logger.info(f"保存 {data_name} 成功!")
    return save_path
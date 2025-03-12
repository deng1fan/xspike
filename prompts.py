from xspike.io import load_in
from transformers import AutoTokenizer
from tqdm.auto import tqdm 

# 全局 tokenizer 缓存
_TOKENIZER_CACHE = {}

def get_tokenizer(model_id, show_progress=True):
    """获取并缓存 tokenizer，避免重复加载
    
    Args:
        model_id: 模型ID或名称
        show_progress: 是否显示加载进度（仅在首次加载时显示）
    
    Returns:
        AutoTokenizer: 加载的tokenizer对象
    """
    if model_id not in _TOKENIZER_CACHE:
        if show_progress:
            print(f"首次加载 {model_id} 的tokenizer...")
        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
        if show_progress:
            print(f"{model_id} tokenizer 加载完成")
    return _TOKENIZER_CACHE[model_id]

# 仅针对没有 apply_chat_template 的特殊模型格式
SPECIAL_FORMATS = {
    "llama2": lambda msgs, sys: format_llama2(msgs, sys),
    "vicuna_psy": lambda msgs, sys: format_vicuna(msgs, sys, "Client", "Counsellor"),
    "vicuna_gen": lambda msgs, sys: format_vicuna(msgs, sys, "USER", "ASSISTANT"),
    "mistral": lambda msgs, sys: format_mistral(msgs, sys),
    "deepseek": lambda msgs, sys: format_deepseek(msgs, sys)
}

def normalize_messages(dials, sys_prompt=None, user_as_last=True):
    """
    将各种格式的对话标准化为符合 HuggingFace chat_template 的消息列表
    
    Args:
        dials: 对话历史，支持字典列表或字符串列表
        sys_prompt: 系统提示，如果提供会添加为系统消息
        user_as_last: 布尔值，处理字符串列表时:
            - True: 确保最后一个消息是用户的(从末尾开始反向分配角色)
            - False: 确保第一个消息是用户的(从开头开始正向分配角色)
    
    Returns:
        list: 标准化的消息列表，格式为 [{"role": "...", "content": "..."}, ...]
    """
    if isinstance(dials[0], dict) and "role" in dials[0] and "content" in dials[0]:
        messages = dials.copy()
    else:
        messages = []
        if user_as_last:
            # 反向遍历确保最后一个是用户消息
            for i, dial in enumerate(reversed(dials)):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": dial})
            messages.reverse()
        else:
            # 正向遍历，第一个是用户消息
            for i, dial in enumerate(dials):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": dial})
    
    # 添加系统提示
    if sys_prompt is not None:
        if len(messages) > 0 and messages[0]["role"] == "system":
            messages[0]["content"] = sys_prompt
        else:
            messages.insert(0, {"role": "system", "content": sys_prompt})
            
    return messages

def apply_prompt(dials, model_id="llama3", sys_prompt=None, tokenize=False, user_as_last=True):
    """
    将对话历史转换为模型输入格式，优先使用 tokenizer.apply_chat_template
    
    这个函数接收对话历史，并将其转换为特定模型所需的输入格式。
    优先尝试使用模型的原生 tokenizer.apply_chat_template 方法，
    当该方法不可用时，会回退到为特定模型实现的手动格式化函数。
    
    Args:
        dials: 对话历史，支持两种格式:
            1. 字典列表: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            2. 字符串列表: ["用户消息1", "助手回复1", "用户消息2", ...]
        model_id: 模型ID或类型名称，用于确定如何格式化输入
        sys_prompt: 系统提示文本，如果提供，将覆盖模型默认的系统提示
        tokenize: 是否返回tokenized结果而非文本字符串
        user_as_last: 布尔值，处理字符串列表时:
            - True: 确保最后一个消息是用户的(从末尾开始反向分配角色)
            - False: 确保第一个消息是用户的(从开头开始正向分配角色)
        
    Returns:
        str或list: 处理后的模型输入，根据tokenize参数返回字符串或token列表
    
    注意:
        当dials为字典列表时，user_as_last参数会被忽略，因为角色已明确指定
    """
    # 标准化消息格式
    messages = normalize_messages(dials, sys_prompt, user_as_last)
    
    # 首先尝试使用 tokenizer 的 apply_chat_template 方法
    try:
        tokenizer = get_tokenizer(model_id, False)  # 不显示进度
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=tokenize
            )
    except Exception as e:
        pass  # 静默失败，转到备用方法
    
    # 仅当 tokenizer 方法不可用时，使用特定模型格式的备用方法
    format_func = SPECIAL_FORMATS.get(model_id.lower())
    if format_func:
        return format_func(messages, sys_prompt or "")
    
    # 对于未处理的特殊情况，尝试基于模型名称猜测格式
    model_lower = model_id.lower()
    if "llama" in model_lower and "3" in model_lower:
        return format_llama3(messages, sys_prompt or "")
    elif "llama" in model_lower:
        return format_llama2(messages, sys_prompt or "")
    elif "qwen" in model_lower:
        return format_qwen(messages, sys_prompt or "")
    
    # 最后尝试再次使用 tokenizer，即使前面失败了
    # 这是因为某些模型可能支持 apply_chat_template 但有特定的错误处理需求
    try:
        tokenizer = get_tokenizer(model_id, False)  # 不显示进度
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=tokenize
        )
    except Exception as e:
        raise ValueError(f"无法为模型 {model_id} 格式化消息: {e}")

def format_llama2(messages, sys_persona):
    """为 Llama2 模型格式化消息"""
    input_str = f"<s>[INST] <<SYS>> {sys_persona} <</SYS>>"
    first_user = True
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            if not first_user:
                input_str += f"<s>[INST] {msg['content']}"
            else:
                input_str += f" {msg['content']}"
                first_user = False
        else:  # assistant
            input_str += f"[/INST] {msg['content']}</s>"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += "[/INST]"
    
    return input_str

def format_llama3(messages, sys_persona):
    """为 Llama3 模型格式化消息"""
    input_str = f"<|start_header_id|>system<|end_header_id|>\n{sys_persona}\n<|eot_id|>"
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            input_str += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>"
        else:  # assistant
            input_str += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += "<|start_header_id|>assistant<|end_header_id|>\n"
    
    return input_str

def format_qwen(messages, sys_persona):
    """为 Qwen 模型格式化消息"""
    input_str = f"<|im_start|>system\n{sys_persona}<|im_end|>\n"
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            input_str += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        else:  # assistant
            input_str += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += "<|im_start|>assistant\n"
    
    return input_str

def format_vicuna(messages, sys_persona, user_token="USER", assistant_token="ASSISTANT"):
    """为 Vicuna 模型格式化消息"""
    input_str = sys_persona
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            input_str += f" {user_token}: {msg['content']}"
        else:  # assistant
            input_str += f" {assistant_token}: {msg['content']}</s>"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += f" {assistant_token}: "
    
    return input_str

def format_mistral(messages, sys_persona):
    """为 Mistral 模型格式化消息"""
    input_str = "<s>" + sys_persona
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            input_str += f"[INST]{msg['content']}"
        else:  # assistant
            input_str += f"[/INST]{msg['content']}</s>"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += "[/INST]"
    
    return input_str

def format_deepseek(messages, sys_persona):
    """为 DeepSeek 模型格式化消息"""
    sys_prefix = "<｜begin▁of▁sentence｜>" if not sys_persona else f"<｜begin▁of▁sentence｜>System:{sys_persona}<｜end▁of▁sentence｜>"
    input_str = sys_prefix
    
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            input_str += f"<｜User｜>{msg['content']}"
        else:  # assistant
            input_str += f"<｜Assistant｜>{msg['content']}<｜end▁of▁sentence｜>"
    
    # 如果最后一个消息是用户，添加模型回复的开始标记
    if messages[-1]["role"] == "user":
        input_str += "<｜Assistant｜>"
    
    return input_str

def to_std_dial(dials, show_progress=True):
    """
    将嵌套对话列表转换为标准格式
    
    将形如 [["用户消息1", "助手回复1", ...], ["用户消息2", ...]] 的嵌套列表
    转换为形如 [{"conversations": [{"role": "user", "content": "用户消息1"}, ...]}, ...]
    的标准格式
    
    Args:
        dials: 嵌套对话列表，每个子列表表示一个完整对话，
               假设每个对话都是用户开始，用户和助手交替发言
        show_progress: 是否显示进度条
    
    Returns:
        list: 标准格式的对话列表，每个对话是一个包含"conversations"键的字典
    """
    std_dial = []
    # 添加进度条
    iterator = tqdm(range(len(dials)), desc="转换对话格式") if show_progress else range(len(dials))
    
    for i in iterator:
        conv = []
        for j in range(len(dials[i])):
            conv.append({
                "role": "user" if j % 2 == 0 else "assistant",
                "content": dials[i][j],
            })
        std_dial.append({"conversations": conv})
    return std_dial

def apply_prompt_pair(dials, model_type="llama3", sys_prompt=None, show_progress=True):
    """
    将对话数据集转换成模型训练所需的输入-输出对。
    
    这个函数处理一个或多个完整对话，并从中提取所有可能的训练样本。
    每个训练样本包含：
    1. 输入(input): 包括系统提示和对话历史直到用户的当前消息，格式化为模型特定的格式
    2. 输出(output): 对应该用户消息的助手回复
    
    例如，对于一个4轮对话 [user1, assistant1, user2, assistant2]，会生成2个训练样本:
    - 样本1: input=格式化的(system + user1), output=assistant1
    - 样本2: input=格式化的(system + user1 + assistant1 + user2), output=assistant2
    
    Args:
        dials: 对话数据，支持三种格式:
            1. 文件路径: 指向包含对话数据的JSONL文件的字符串
            2. 嵌套列表: 形如 [["用户消息1", "助手回复1", "用户消息2", ...], [...]]
            3. 字典列表: 形如 [{"conversations": [{"role": "user", "content": "..."}, ...]}, {...}]
        model_type: 模型类型或ID，用于确定如何格式化输入
        sys_prompt: 系统提示内容，如果提供，将覆盖模型默认的系统提示
        show_progress: 是否显示进度条
        
    Returns:
        list: 包含训练样本的字典列表，每个字典有两个键:
            - "input": 格式化的模型输入，包括对话历史直到用户消息
            - "output": 期望的模型输出，即助手的回复
    
    示例:
        >>> dials = [["你好", "你好，我能帮你什么?", "讲个笑话", "为什么程序员总是混淆万圣节和圣诞节？因为 Oct 31 == Dec 25"]]
        >>> samples = apply_prompt_pair(dials, model_type="llama3")
        >>> print(len(samples))
        2
        >>> print(samples[0]["output"])
        你好，我能帮你什么?
    """
    # 处理输入数据，支持三种格式
    if isinstance(dials, str):
        # 1. 文件路径：加载JSONL文件
        ori_inputs = load_in(dials)
    else:
        ori_inputs = dials
        
    # 转换为标准格式
    if not isinstance(ori_inputs[0], dict):
        # 2. 嵌套列表：转换为标准格式的字典列表
        ori_inputs = to_std_dial(ori_inputs)
    elif "conversations" not in ori_inputs[0]:
        raise ValueError("无效的输入格式：缺少'conversations'键，请确保每个对话都包含有效的对话轮次")
    
    formatted_inputs = []
    # 遍历每个对话，添加进度条
    conversations_iterator = tqdm(ori_inputs, desc="处理对话") if show_progress else ori_inputs
    
    for conv_data in conversations_iterator:
        turns = conv_data["conversations"]
        
        # 遍历对话中的每个用户消息
        for i in range(0, len(turns) - 1):
            # 确保当前是用户消息，下一个是助手消息
            if turns[i]["role"] != "user" or turns[i+1]["role"] != "assistant":
                continue
                
            # 提取当前对话历史，包括当前用户消息
            current_messages = turns[:i+1]
            
            # 使用apply_prompt生成模型所需的格式化输入
            full_prompt = apply_prompt(
                current_messages,
                model_id=model_type,
                sys_prompt=sys_prompt,
                tokenize=False
            )
            
            # 创建训练样本并添加到结果
            formatted_inputs.append({
                "input": full_prompt,          # 格式化的输入（包括系统提示和对话历史）
                "output": turns[i + 1]["content"]  # 期望的输出（助手回复）
            })
    
    return formatted_inputs
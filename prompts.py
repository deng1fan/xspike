from xspike.io import load_in

class Llama2Prompt:
    def __init__(self):
        self.sys_persona = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal data. "
            "Please ensure that your responses are socially unbiased and positive in nature. "
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct."
            "If you don't know the answer to a question, please don't share false information."
            )
        self.user_start_token = "<s>[INST]"
        self.bot_start_token = "[/INST]"
        self.eos_token = "</s>"


    def build_input(self, history=[], user_input=""):
        input_str = (
            f"<s>[INST] <<SYS>> {self.sys_persona} <</SYS>>"
        )
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token if idx != 0 else ""
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"]
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        user_perfix = self.user_start_token if len(history) != 0 else ""
        input_str += user_perfix + user_input + self.bot_start_token
        return input_str
    
    
class QWen2Prompt:
    def __init__(self):
        self.sys_persona = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.user_start_token = "<|im_start|>user\n"
        self.bot_start_token = "<|im_start|>assistant\n"
        self.eos_token = "<|im_end|>\n"
        

    def build_input(self, history=[], user_input=""):
        input_str = (
            f"<|im_start|>system\n{self.sys_persona}<|im_end|>\n"
        )
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"] + self.eos_token
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        user_perfix = self.user_start_token
        input_str += user_perfix + user_input + self.eos_token + self.bot_start_token
        return input_str
    
    
    
class QWen2PrismPrompt:
    def __init__(self):
        self.sys_persona = "你是一个数据合成器，负责根据以下示例生成新的派生数据。新的数据应保留与原始示例相似的结构和核心内容，但要在细节上进行适当变化，确保生成的数据需与示例有足够的区别，以避免重复，同时确保内容合乎逻辑且真实可信。生成的内容必须始终围绕指定的主题或话题，并严格遵循要求。每条数据都对应着一条唯一的序列号，序列号的不同将对新数据的内容产生影响。"
        self.user_start_token = "<|im_start|>user\n"
        self.bot_start_token = "<|im_start|>assistant\n"
        self.eos_token = "<|im_end|>\n"
        

    def build_input(self, history=[], user_input=""):
        input_str = (
            f"<|im_start|>system\n{self.sys_persona}<|im_end|>\n"
        )
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"] + self.eos_token
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        user_perfix = self.user_start_token
        input_str += user_perfix + user_input + self.eos_token + self.bot_start_token
        return input_str
    
    
    
class Llama3Prompt:
    def __init__(self):
        self.sys_persona = ""
        self.user_start_token = "<|start_header_id|>user<|end_header_id|>\n"
        self.bot_start_token = "<|start_header_id|>assistant<|end_header_id|>\n"
        self.eos_token = "<|eot_id|>"

    def build_input(self, history=[], user_input=""):
        input_str = (
            f"<|start_header_id|>system<|end_header_id|>\n{self.sys_persona}\n<|eot_id|>"
        )
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"] + self.eos_token
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        user_perfix = self.user_start_token
        input_str += user_perfix + user_input + self.eos_token + self.bot_start_token
        return input_str

    
    
class VicunaPsyPrompt:
    def __init__(self):
        self.sys_persona = "you are the role of a professional counsellor who specialises in using a wide range of counselling techniques to provide deep guidance and insight. Avoid coaching responses and provide tailored advice to help the client based on the client's feedback."
        self.user_start_token = "Client: "
        self.bot_start_token = "Counsellor: "
        self.eos_token = "</s>"
        

    def build_input(self, history=[], user_input=""):
        input_str = self.sys_persona
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += " " + user_perfix + turn["text"]
            else:
                input_str += " " + bot_perfix + turn["text"] + self.eos_token
        input_str += " " + self.user_start_token + user_input + " " +  self.bot_start_token
        return input_str



class VicunaGenPrompt:
    def __init__(self):
        self.sys_persona = "You are a data synthesizer. Generating rewritten data that follow the format and content of the demo but include some variations to ensure the new data is diverse and realistic."
        self.user_start_token = "USER: "
        self.bot_start_token = "ASSISTANT: "
        self.eos_token = "</s>"
        

    def build_input(self, history=[], user_input=""):
        input_str = self.sys_persona
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += " " + user_perfix + turn["text"]
            else:
                input_str += " " + bot_perfix + turn["text"] + self.eos_token
        input_str += " " + self.user_start_token + user_input + " " +  self.bot_start_token
        return input_str





class MistralPrompt:
    def __init__(self):
        self.sys_persona = ""
        self.user_start_token = "[INST]"
        self.bot_start_token = "[/INST]"
        self.eos_token = "</s>"

    def build_input(self, history=[], user_input=""):
        input_str = self.sys_persona
        for idx, turn in enumerate(history):
            user_perfix = self.user_start_token
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"]
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        input_str += self.user_start_token + user_input + self.bot_start_token
        return "<s>" + input_str


class DeepSeekPrompt:
    def __init__(self):
        self.sys_persona = ""
        self.user_start_token = "<｜User｜>"
        self.bot_start_token = "<｜Assistant｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"

    def build_input(self, history=[], user_input=""):
        input_str = "<｜begin▁of▁sentence｜>" if self.sys_persona == "" else "<｜begin▁of▁sentence｜>System:" + self.sys_persona + self.eos_token
        user_perfix = self.user_start_token
        for idx, turn in enumerate(history):
            bot_perfix = self.bot_start_token
            if turn["speaker"] == "user":
                input_str += user_perfix + turn["text"]
            else:
                input_str += bot_perfix + turn["text"] + self.eos_token
        input_str += user_perfix + user_input + self.bot_start_token
        return input_str
    


PROMPT_DICT = {
    "llama2": Llama2Prompt(),
    "vicuna_psy": VicunaPsyPrompt(),
    "vicuna_gen": VicunaGenPrompt(),
    "mistral": MistralPrompt(),
    "llama3": Llama3Prompt(),
    "qwen2": QWen2Prompt(),
    "qwen_prism": QWen2PrismPrompt(),
    "seepseek": DeepSeekPrompt(),
}



def to_std_dial(dials):
    std_dial = []
    for i in range(len(dials)):
        conv = []
        for j in range(len(dials[i])):
            conv.append(
                {
                    "role": "user" if j % 2 == 0 else "assistant",
                    "content": dials[i][j],
                }
            )
        std_dial.append({"conversations": conv})
    return std_dial



def apply_prompt_pair(
    dials, model_type="qwen2", sys_prompt=None
):
    """
    将对话数据集转换成模型输入格式
    dials Format 1:
    {
        "conversations": [
            {
                "role": "user",
                "content": "Hello, how can I help you?"
            },
            {
                "role": "assistant",
                "content": "I'm sorry, I don't have the answer to your question."
            },
            ...
        ]
    }
    {
        "conversations": [
            {
                "role": "user",
                "content": "Hello!"
            },
            {
                "role": "assistant",
                "content": "I'm sorry."
            },
            ...
    }
    ......

    dials Format 2:
    [
        ["Hello, how can I help you?", "I'm sorry, I don't have the answer to your question."],
        ["Hello!", "I'm sorry."]
        ......
    ]
    
    dials Format 3:
    "/path/to/dials.jsonl"

    Args:
        dials: JSONL file containing conversation data or list of conversation data
        model_type: One of the supported model types (llama2, vicuna_psy, etc.)

    Returns:
        List of formatted input strings ready for model consumption
    """
    # Get the appropriate prompt builder
    prompt_builder = PROMPT_DICT.get(model_type, None)
    if not prompt_builder:
        raise ValueError(f"Unsupported model type: {model_type}")

    if sys_prompt:
        prompt_builder.sys_persona = sys_prompt
        
    # Read and process the JSONL file
    if isinstance(dials, str):
        ori_inputs = load_in(dials)
    else:
        ori_inputs = dials
        
    if not isinstance(ori_inputs[0], dict):
        ori_inputs = to_std_dial(ori_inputs)
    elif "conversations" not in ori_inputs[0]:
        raise ValueError("Invalid input format, 'conversations' key not found")
        
    
    
    formatted_inputs = []
    for conv_data in ori_inputs:
        turns = conv_data["conversations"]

        # 准备对话历史缓存
        dialog_history = []

        # 遍历所有对话轮次，构建多轮输入
        for i in range(0, len(turns), 2):
            # 确保对话的最后一句话以 assistant 结束
            if i + 1 >= len(turns) and turns[i]["role"] != "assistant":
                break

            # 构建历史对话结构
            history = [
                {
                    "speaker": "user" if t["role"] == "user" else "assistant",
                    "text": t["content"],
                }
                for t in turns[:i]
            ]

            # 当前用户输入
            user_input = turns[i]["content"]
            # 模型输出
            sys_output = turns[i + 1]["content"] + prompt_builder.eos_token

            # 构建完整 prompt
            full_prompt = prompt_builder.build_input(
                history=history, user_input=user_input
            )

            formatted_inputs.append({"input": full_prompt, "output": sys_output})

    return formatted_inputs


def apply_prompt(dials, model_type="qwen2", sys_prompt=None):
    """
    将对话历史转换为模型输入格式，dials 的最后一句话以 user 结束
    dials Format 1:
    [
            {
                "role": "user",
                "content": "Hello, how can I help you?"
            },
            {
                "role": "assistant",
                "content": "I'm sorry, I don't have the answer to your question."
            },
            ...
    ]

    dials Format 2:
    [
        ["Hello, how can I help you?", "I'm sorry, I don't have the answer to your question."],
        ["Hello!", "I'm sorry."]
        ......
    ]
    """
    # 获取对应的prompt构建器
    prompt_builder = PROMPT_DICT.get(model_type)
    if not prompt_builder:
        raise ValueError(f"Unsupported model type: {model_type}")

    if sys_prompt:
        prompt_builder.sys_persona = sys_prompt

    # 标准化输入格式
    if isinstance(dials[0], dict):
        # 处理带有"conversations"键的字典格式
        # 假设是单轮对话字典列表（如直接传递对话轮次）
        # 需确保每个字典有role和content
        if not all("role" in item and "content" in item for item in dials):
            raise ValueError("无效的对话格式：确保每个字典都有role和content键")
    else:
        # 处理列表的列表（如[["A", "B"], ["C"]])，转为标准对话格式
        dials = to_std_dial([dials])[0]['conversations']

    if dials[-1]["role"] != "user":
        raise ValueError("Last turn must be user input")
    # 检查是否是 user 和 assistant 交替的对话，可以是 user 开始，也可以是 assistant 开始
    prev_role = None
    for d in dials:
        current_role = d["role"]
        if prev_role == current_role:
            raise ValueError("无效的对话格式：确保对话轮次交替进行")
        prev_role = current_role

    # 构建历史对话和当前输入
    history = [
        {"speaker": "user" if t["role"] == "user" else "assistant", "text": t["content"]}
        for t in dials[:-1]
    ]
    user_input = dials[-1]["content"]

    # 生成完整prompt
    full_prompt = prompt_builder.build_input(history=history, user_input=user_input)


    return full_prompt 
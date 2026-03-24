
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name_or_path, use_flash_attn=True, custom_configs={}):
    if "Qwen" in model_name_or_path:
        custom_configs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
           
            "dtype": torch.float16,
        }
        
    if(custom_configs):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **custom_configs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        
    print("Using FP16 and normal attention implementation...")

    tokenizer = load_tokenizer(model_name_or_path)
    
    return model, tokenizer
def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = '<|endoftext|>'
            tokenizer.pad_token_id = tokenizer.eod_id
    if tokenizer.clean_up_tokenization_spaces:
        print(
            "WARNING: tokenizer.clean_up_tokenization_spaces is by default set to True. "
            "This will break the attack when validating re-tokenization invariance. Setting it to False..."
        )
        tokenizer.clean_up_tokenization_spaces = False

    # If the chat template is not available, manually set one
    # Some older models do not come with a chat template in their tokenizer
    # Alternatively, you can also modify `tokenizer_config.json` file in the model directory, if you locally have access to it
    if "Qwen/" in model_name_or_path:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token 
    if not tokenizer.chat_template:
        print(
            f"The tokenizer of {model_name_or_path} does not come with a chat template. Dynamically setting one..."
        )
        if "harmbench-llama2-13b-cls" in model_name_or_path:
            # If you are sure that the model does not require a chat template, you can skip this step like this
            print(
                "HarmBench-Llama-2-13b-cls does not require a chat template. Skipped."
            )
        elif "vicuna-7b-v1.5" in model_name_or_path:
            # Otherwise, please implement the chat template manually in Jinja language like this
            # No indentation and newlines are allowed. Please make it a single line
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"
        elif "Qwen" in model_name_or_path:       
            tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        elif "baichuan" in model_name_or_path:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<reserved_106>' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<reserved_107>' + message['content'] + '</s>'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<reserved_107>' }}{% endif %}"
        elif "llama2-70b" in model_name_or_path:
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
        else:
            raise ValueError(
                f"The chat template for the tokenizer of {model_name_or_path} is not available. "
                "To avoid unexpected behavior, it cannot proceed with the default chat template. "
                "Please implement it manually in `load_model_and_tokenizer()`, `utils.py`."
            )
    return tokenizer


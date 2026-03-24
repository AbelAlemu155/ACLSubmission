import time
import torch,re

from debate_codes.opensource.util.load_models import load_model_and_tokenizer

# def query_one_model(model_paths, batch_messages, batch_prompts= None,
#  rounds= 0):

#     model_path= model_paths[0]
#     """
#     Run a batch of messages through a model and return responses with time and cost.

#     Args:
#         model_path (str): Path to the model.
#         batch_messages (list): Each element is a multi-turn message history, e.g.,
#             [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "..."}]

#     Returns:
#         list of dict: Each dict contains:
#             - response (str)
#             - tokens_generated (int)
#             - time_taken (float)
#             - cost_usd (float)
#     """
#     # Custom configurations for Qwen
#     custom_configs = {}
#     if "Qwen" in model_path:
#         custom_configs = {
#             "device_map": "auto",
#             "low_cpu_mem_usage": True,
#             "torch_dtype": torch.float16,
#             "trust_remote_code": True,
#             "enable_thinking": True,
#             "dtype": torch.float16,
#         }
#         custom_gen_config={
#             "temperature": 0.6,
#             "top_p": 0.95,
#             "top_k": 20,
#             "min_p": 0
#         }

#     # Load model and tokenizer
#     model, tokenizer = load_model_and_tokenizer(model_path, custom_configs)
#     # model.eval()

#     # Convert multi-turn messages to batch prompts using tokenizer's chat template
#     # batch_prompts = [tokenizer.apply_chat_template(message_history) for message_history in batch_messages]
#     batch_prompts = [
#     tokenizer.apply_chat_template(
#         message_history,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     for message_history in batch_messages
#     ]
#     # print(batch_prompts)
#     # Tokenize batch prompts
#     # print(f"first step: {batch_prompts}")
#     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     # print(f"second step: {inputs}: length: {len(inputs)}")
#     # Generate responses for the batch
#     start_time = time.time()
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             do_sample=True,
#             use_cache=True, 
#             temperature=custom_gen_config.get("temperature", 1.0),
#             top_p=custom_gen_config.get("top_p", 1.0),
#             top_k=custom_gen_config.get("top_k", 50)

#         )
   

#     # print("third step")
#     time_taken = time.time() - start_time

#     # Decode only newly generated tokens
#     batch_responses = tokenizer.batch_decode(
#         outputs[:, inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )
#     # print(f"batch_responses {batch_responses}")
#     # Calculate tokens and cost for each response
#     results = []
#     total_cost=0
#     for response_text, prompt in zip(batch_responses, batch_prompts):
#         prompt_tokens = len(tokenizer(prompt)["input_ids"])
#         response_tokens = len(tokenizer(response_text)["input_ids"])
#         total_tokens = prompt_tokens + response_tokens

#         # Cost calculation
#         if "Qwen" in model_path:
#             cost_usd = total_tokens / 1_000_000 * 3.5
#         elif "Deepseek" in model_path:
#             cost_usd = total_tokens / 1_000_000 * 2.5
#         else:
#             cost_usd = 0.0

#         results.append(response_text)
#         total_cost+=cost_usd
        

#     return results, time_taken, total_cost



def query_one_model(model_objects, batch_messages, batch_prompts= None,
 rounds= 0):

    # model_path= model_paths[0]
    """
    Run a batch of messages through a model and return responses with time and cost.

    Args:
        model_path (str): Path to the model.
        batch_messages (list): Each element is a multi-turn message history, e.g.,
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "..."}]

    Returns:
        list of dict: Each dict contains:
            - response (str)
            - tokens_generated (int)
            - time_taken (float)
            - cost_usd (float)
    """

    # Custom configurations for Qwen
    # custom_configs = {}
    # if "Qwen" in model_path:
    #     custom_configs = {
    #         "device_map": "auto",
    #         "low_cpu_mem_usage": True,
    #         "torch_dtype": torch.float16,
    #         "trust_remote_code": True,
    #         "enable_thinking": True,
    #         "dtype": torch.float16,
    #     }
    #     custom_gen_config={
    #         "temperature": 0.6,
    #         "top_p": 0.95,
    #         "top_k": 20,
    #         "min_p": 0
    #     }

    # # Load model and tokenizer
    # model, tokenizer = load_model_and_tokenizer(model_path, custom_configs)
    # # model.eval()
    
    model = model_objects[0]['model']
    model_path= model.config._name_or_path
    tokenizer= model_objects[0]['tokenizer']
    # Convert multi-turn messages to batch prompts using tokenizer's chat template
    # batch_prompts = [tokenizer.apply_chat_template(message_history) for message_history in batch_messages]
    if("Qwen" in model_path):
        batch_prompts = [
        tokenizer.apply_chat_template(
            message_history,
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=False
        )
        for message_history in batch_messages
        ]
    else:
        batch_prompts = [
        tokenizer.apply_chat_template(
            message_history,
            tokenize=False,
            add_generation_prompt=True, 
        )
        for message_history in batch_messages
        ]

    # print(batch_prompts)
    # Tokenize batch prompts
    # print(f"first step: {batch_prompts}")
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # print(f"second step: {inputs}: length: {len(inputs)}")
    # Generate responses for the batch
    start_time = time.time()
    
    custom_gen_config={}
    if ("Qwen" in model_path):
        custom_gen_config={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0
        }
    
    with torch.no_grad():
        if(custom_gen_config):

            outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    use_cache=True, 
                    
                    temperature=custom_gen_config.get("temperature", 1.0),
                    top_p=custom_gen_config.get("top_p", 1.0),
                    top_k=custom_gen_config.get("top_k", 50))
        else: 
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                use_cache=True, 
                temperature=custom_gen_config.get("temperature", 1.0),
                top_p=custom_gen_config.get("top_p", 1.0),
                top_k=custom_gen_config.get("top_k", 50)
            )
    
   

    # print("third step")
    time_taken = time.time() - start_time

    # Decode only newly generated tokens
    batch_responses = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    # print(batch_responses)
    # print(f"batch_responses {batch_responses}")
    # Calculate tokens and cost for each response
    results = []
    total_cost=0
    for response_text, prompt in zip(batch_responses, batch_prompts):
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        response_tokens = len(tokenizer(response_text)["input_ids"])
        total_tokens = prompt_tokens + response_tokens

        # Cost calculation
        if "Qwen" in model_path:
            cost_usd = total_tokens / 1_000_000 * 3.5
        elif "Deepseek" in model_path:
            cost_usd = total_tokens / 1_000_000 * 2.5
        else:
            cost_usd = 0.0
        
        clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

        clean_text = clean_text.strip()
        results.append(clean_text)
        total_cost+=cost_usd
        

    return results, time_taken, total_cost




if __name__== "__main__":
    # {"role": "system", "content":"You are an assitant"}
    batch_messages= [[{"role": "system", "content": "You are an assistant"},{"role": "user", "content": "What is your version?"}] for _ in range(8)]
    # batch_messages= ["what is your name"]
    # model_paths= ["Qwen/Qwen3-8B"]
    model_paths= ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]
    print(query_one_model(model_paths, batch_messages))
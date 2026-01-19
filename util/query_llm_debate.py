from util.query_llm import query_llm

def query_llm_debate(client, model_configs, prompt, max_tokens, role, fs_examples=None, temperature=0, max_retries=8,rounds= 2):

    best_outputs = []
    model_histories = {}
    model_config_map= {}
    for model_config in model_configs:
        model_name = model_config['name']
        model_config_map[model_name]= model_config
        model_histories[model_name] = [

            #  reasoning prompting techniques might be effective here 
            #  reasoning with few shots 
            #  reasoning with chain of thoughts 
            # 
            {"role": "system", "content": role}, 
            {"role": "user", "content": prompt}
          ]
        

    total_time=0
    total_cost=0
    for r in range(1, rounds + 1):
        round_outputs = {}
        for model_config in model_configs:

            model_name = model_config['name']
            curr_prompt =''
            if best_outputs:
                
                for other_model, other_output in best_outputs[-1].items():
                    if other_model != model_name:
                       
                        curr_prompt =  f"[{other_model.upper()} output from previous round]: {other_output}"
            if(curr_prompt != ''):
                curr_prompt = curr_prompt + "\n Refine your answer considering different round outputs. Choose the points judiciously choose the correct points from other models. Carefully analyze other model outputs and provide answer concisely."
                model_histories[model_name].append({"role": "user", "content": curr_prompt})
            # model_histories[model_name].extend(messages)
            output, time, cost = query_llm(client, [model_config], None, max_tokens, role,un_to_comply=[],  messages=model_histories[model_name],)
            total_time+=time
            total_cost += cost
            model_histories[model_name].append({"role": "assistant", "content": output})

            round_outputs[model_name] = output

        best_outputs.append(round_outputs)

    final_message = best_outputs[-1][model_configs[0]['name']]
    return final_message, total_time, total_cost 




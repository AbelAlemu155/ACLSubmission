# child_dir/script.py
from ..prompts import cri_prompt, judge_prompt
from .query_model import query_one_model  
def critic_with_judge(model_objects,batch_messages,batch_prompts, rounds=1 ):
    
    # reasoning_model, critic_model, judge_model = model_objects 
    reasoning_model= model_objects[0]
    msg_history_b= batch_messages
    total_time =0
    total_cost =0
    for _ in range(rounds):
        response_list, time_taken, cost= query_one_model([reasoning_model], msg_history_b)
        # batch_prompts_cri= []
        total_time+= time_taken 
        total_cost+=cost 

        new_msg_b= [[] for _ in range(len(batch_messages))]
        crit_outputs= []
        for ind,resp in enumerate(response_list):
            new_msg_b[ind].append({"role": "system", "content": "You are a critic."})
            new_msg_b[ind].append({"role": "user", "content": cri_prompt.format(prompt=batch_prompts[ind], output= resp)})      
            crit_outputs.append(resp)
        
        response_list, time_taken, cost = query_one_model([reasoning_model], new_msg_b)
        
        new_msg_b= [[] for _ in range(len(batch_messages))]
        total_time+=time_taken
        total_cost+=cost
        for ind,resp in enumerate(response_list):
            new_msg_b[ind].append({"role": "system", "content": "You are an evidence arbiter."})
            new_msg_b[ind].append({"role": "user", "content": judge_prompt.format(prompt=batch_prompts[ind], output= resp, crit_output= crit_outputs[ind])})      
        
        response_list, time_taken, cost = query_one_model([reasoning_model], new_msg_b)
        total_time+=time_taken
        total_cost+=cost  

 
    return response_list, total_time, total_cost


if __name__=="__main__":
    batch_messages= [[{"role": "system", "content": "You are an assistant"},{"role": "user", "content": "What is your version?"}] for _ in range(8)]
    # batch_messages= ["what is your name"]
    batch_prompts= ["What is your version?" for _ in range(8)]

    model_paths= ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B"]
    print(critic_with_judge(model_paths, batch_messages, batch_prompts))



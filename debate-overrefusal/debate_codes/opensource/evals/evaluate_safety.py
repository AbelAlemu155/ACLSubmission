
from ..prompts import safety_eval_rubric, safety_eval_prompt
from openai import OpenAI
import os, re, random, time
from tqdm import tqdm
from openai import BadRequestError
from ..util.query_model import query_one_model
from ..util.critic_debate_open_source import critic_with_judge
import json

from ..util.build_model_objects import build_model_objects
from debate_codes.data_processing import read_compliance, read_compliance_law
# from ..util.query_model import query_one_model
from ..util.construct_message import construct_message
def evaluate_safety_gpt(question, generated_response, safe_response, num_iter_val=1,max_retries=8):
    total_score= 0
    client = OpenAI(api_key = os.getenv("OPEN_AI_API_KEY")) 
    messages= construct_message(system="You are a system assistant",

    prompt=safety_eval_prompt.format(question=question,generated_response=generated_response , 
                                        safe_response=safe_response, rubric= safety_eval_rubric ))


    for attempt in range(max_retries):
        try:
            response = client.responses.create(
            model="gpt-4.1",
            input=messages, 
            temperature=0,
            top_p=1
            )

            match = re.search(r'#score\s*:\s*([0-9]+(?:\.[0-9]+)?)', response.output_text)
            # score = float(response)
            score = float(match.group(1)) if match else None
            if score is None: 
                score= 2
                print(f"score returned noone")
            total_score += score
            if score < 1 or score > 5:
                raise ValueError(f"Invalid score {score}")
            score = total_score/ num_iter_val
            return score, response
        except BadRequestError as e:
            error_msg = str(e)
    
            if "ContentPolicyViolation" in error_msg or "filtered" in error_msg:
               
                # Treat as compliant refusal
                return 'Unable to comply with the request', 0, 0
       
        except Exception as e:
            # Generic catch for rate-limiting or transient errors
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.random()
                print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Re-raise other errors
                print(e)
                raise e

def evaluate_trident(compliance_df, evaluator, model_paths, output_file="trident.json", metric_file="metric_trident.json", num_iter_eval=1, temperature=0, batch_size=4):
    results = []
    total_time =0
    total_cost=0     
    total_score=0
    model_objects= build_model_objects(model_paths)
    for start in tqdm(range(0, len(compliance_df), batch_size)):
        batch_prompts = []    
        batch_messages=[]
        batch_safe_resp= []
        batch_df = compliance_df.iloc[start:start+batch_size]
     
        for ind, row in batch_df.iterrows():
            harmful_prompt = row['harmful_prompt']
            safe_response = row['safe_reply']
            prompt = (
                f"Question:\n{harmful_prompt}\n\n"
            )
            system_prompt =""

            full_prompt = system_prompt + "\n\n" + prompt
            batch_prompts.append(full_prompt)
            batch_messages.append(construct_message(system_prompt, prompt))
            batch_safe_resp.append(safe_response)


        scores = [0 for _ in range(batch_size)]
        score_reasons=[]
        # Generate 3 outputs per query  
        cur_time , cur_cost=0, 0   
        
        for _ in range(num_iter_eval):
            responses, est_time, cost  = evaluator(model_objects, batch_messages, batch_prompts= batch_prompts)
            cur_time += est_time
            cur_cost += cost 
            for ind,resp in enumerate(responses):
                score,reason= evaluate_safety_gpt(batch_prompts[ind], 
                generated_response= resp,safe_response= batch_safe_resp[ind]
                )
                scores[ind]+=score 
                score_reasons.append(reason)  
         
        cur_avg_time= cur_time/num_iter_eval
        cur_avg_cost= cur_cost/num_iter_eval
        total_time += cur_avg_time
        total_cost += cur_avg_cost
        scores = [sc/num_iter_eval for sc in scores]
        total_score +=sum(scores)
        for ind, resp in enumerate(responses) :
            
            result = {
            "harmful_prompt": batch_prompts[ind],
            "generated_response": resp,
            "compliance_score": scores[ind], 
            "reasoning_score": score_reasons[ind].output_text
            }       

            results.append(result)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_score = total_score/len(compliance_df)
    avg_time = total_time /len(compliance_df)
    avg_cost = total_cost/len(compliance_df)
    print(f"Saved {len(results)} entries to {output_file}")
    print(f"Average score: {avg_score}")
    print(f"Averge time: {avg_time}")
    print(f"Average cost: {avg_cost}")

    metric_results = {'av_score': avg_score, 'av_time': avg_time, 'av_cost': avg_cost}
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump([metric_results], f, ensure_ascii=False, indent=2)


if __name__=="__main__":
    compliance_df= read_compliance()
    compliance_df_law=read_compliance_law()
    # model_paths= ["Qwen/Qwen3-8B"]
    # model_paths=["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]
    # model_paths= ["mistralai/Mistral-7B-Instruct-v0.2"]
    # evaluate_trident(compliance_df_law, query_one_model,
    #                   model_paths, output_file="mistral-trid-law.json", metric_file="mistral-trid-law-metric.json", batch_size=8)   


    model_paths = ["Qwen/Qwen3-8B"]
    evaluate_trident(compliance_df, critic_with_judge,
                      model_paths, output_file="Qwen-critic-trid-med.json", metric_file="Qwen-critic-trid-law-metric.json", batch_size=8)   

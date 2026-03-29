from tqdm import tqdm
from collections import Counter
import re,json
from debate_codes.opensource.util.construct_message import construct_message

from debate_codes.opensource.util.query_model import query_one_model, query_log_prob
from debate_codes.data_processing import read_pubmed
from ..util.build_model_objects import build_model_objects
def evaluate_pubmed_legal(df, model_paths,evaluator,batch_constructor,num_iter_eval = 1,
 batch_size=8, file_path="pubmed.json"):
    final_answers = []
    total_time =0
    total_cost=0
    model_objects= build_model_objects(model_paths)
    for start in tqdm(range(0, len(df), batch_size)):
        
        batch_prompts = []    
        batch_messages=[]
        batch_df = df.iloc[start:start+batch_size]
        for ind, row in batch_df.iterrows():
            query = row['QUESTION']
            context = row['CONTEXTS']
            labels = row['LABELS']
            
            context_label = get_context_with_labels(context, labels)
            # "Think step-by-step internally to reach the correct conclusion.\n"
            # "All reasoning must be grounded exclusively in the provided context.\n"
            system_prompt = (
                "Carefully analyze the question using the provided contexts from a research paper.\n"
                "An explanation detailing the key evidences as Explanation:\n"
                "Provide the answer with only yes/no/maybe as Final answer: yes/no/maybe"
                f"Contexts:\n{context_label}\n\n"
            )

            prompt = (
                f"Question:\n{query}\n\n"
                "Final answer: [Yes|No|Maybe]"
            )

            full_prompt = system_prompt + "\n\n" + prompt
            batch_prompts.append(full_prompt)
            batch_messages.append(construct_message(system_prompt, prompt))

        answers = [[] for _ in range(batch_size)]
        options = [" yes", " no", " maybe"]
        log_prob_batches= [{opt.strip(): 0 for opt in options} for _ in range(len(batch_messages))]
        # Generate 3 outputs per query
        cur_time , cur_cost=0, 0
        for _ in range(num_iter_eval):
            _,lpb, est_time, cost  = evaluator(model_objects, batch_messages, batch_prompts= batch_prompts, rounds= round)
            cur_time += est_time
            cur_cost += cost 
            for ind,lprob in enumerate(lpb):
                for k,v in lprob.items():
                    log_prob_batches[ind][k] += v  
        # log_prob_batches = [{k: v/num_iter_eval for k,v in lprob.items()}  for lprob in log_prob_batches] 
        for lprob in log_prob_batches:
            max_key= max(lprob, key=lprob.get) 
            final_answers.append(max_key)     
        cur_avg_time= cur_time/num_iter_eval
        cur_avg_cost= cur_cost/num_iter_eval
        total_time += cur_avg_time
        total_cost += cur_avg_cost
       
        
    avg_time = total_time/len(df)
    avg_cost = total_cost/len(df)
    print(f"final answers: {final_answers}")
    save_results(final_answers,avg_time, avg_cost, df, file_path)


def evaluate_multi(generated_ans, correct_ans):
    
    matches = sum(1 for gen, corr in zip(generated_ans, correct_ans) if gen == corr)
    accuracy = matches / len(correct_ans)
    return accuracy 



def save_results(pubmed_answers,latency, cost, df, file_path):
 
    results={}
    accuracy = evaluate_multi(pubmed_answers, df.iloc[:len(pubmed_answers)]['final_decision'].tolist())
    results['answers']= pubmed_answers
    results['latency']= latency
    results['cost']=cost
    results['accuracy']=accuracy

    print(f"Accuracy: {accuracy}")
    print(f"cost: {cost}")
    print(f"latency: {latency}")
    try:
        with open(file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
    
    except IOError as e:
        print(f"Error writing to file : {e}")

if __name__=="__main__": 
    df= read_pubmed()
    deeps_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    qwen_path= "Qwen/Qwen3-8B"
    model_paths=[qwen_path]
    evaluate_pubmed_legal(df, model_paths,query_log_prob, file_path="qwen_pubmed.json")
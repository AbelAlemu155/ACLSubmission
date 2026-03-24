from tqdm import tqdm
from collections import Counter
import re
from util.construct_message import construct_message
from debate_codes
def evaluate_pubmed(pubmed_df, model_paths,evaluator, num_iter_eval = 1, batch_size=8):
    final_answers = []
    total_time =0
    total_cost=0

    all_batches = []

    for start in tqdm(range(0, len(pubmed_df), batch_size)):
        
        batch_prompts = []    
        batch_messages=[]
        batch_df = pubmed_df.iloc[start:start+batch_size]
        for ind, row in batch_df.iterrows():
            query = row['QUESTION']
            context = row['CONTEXTS']
            labels = row['LABELS']
            
            context_label = get_context_with_labels(context, labels)

            system_prompt = (
                "Carefully analyze the question using the provided contexts from a research paper.\n"
                "Think step-by-step internally to reach the correct conclusion.\n"
                "All reasoning must be grounded exclusively in the provided context.\n"
                "An explanation detailing the key evidences as Explanation:\n"
                "Provide the answer with only yes/no/maybe as Final answer: yes/no/maybe"
            )

            prompt = (
                f"Question:\n{query}\n\n"
                f"Contexts:\n{context_label}\n\n"
            )

            full_prompt = system_prompt + "\n\n" + prompt
            batch_prompts.append(full_prompt)
            batch_messages.append(construct_message(system_prompt, prompt))

        answers = [[] for _ in range(batch_size)]

        # Generate 3 outputs per query
        cur_time , cur_cost=0, 0
        for _ in range(num_iter_eval):
            responses, est_time, cost  = evaluator(model_paths, batch_messages, batch_prompts= batch_promt, rounds= rounds)
            cur_time += est_time
            cur_cost += cost 
            for ind,resp in enumerate(responses):
                pattern = re.compile(r'(?i)\b(yes|no|maybe)\b')
                match = pattern.search(resp)
                if match:
                    answer = match.group(1).lower()
                    answers[ind].append(answer)
        cur_avg_time= cur_time/num_iter_eval
        cur_avg_cost= cur_cost/num_iter_eval
        total_time += cur_avg_time
        total_cost += cur_avg_cost
        for ans in answers:
           most_common = Counter(ans).most_common(1)[0][0] if ans else "unknown"
           final_answers.append(most_common)
    avg_time = total_time/len(pubmed_df)
    avg_cost = total_cost/len(pubmed_df)
    return final_answers, avg_time, avg_cost
def get_context_with_labels(context, labels):
      lines = context.split('\n')
      # Remove empty lines and strip spaces
      contexts = [line.strip() for line in lines if line.strip()]
      label_list= labels.split(',')
      context_label_list =[f'{label_list[i]}:{contexts[i]}' for i in range(len(contexts))]
      return "/n".join(context_label_list)


def evaluate_multi(generated_ans, correct_ans):
    
    matches = sum(1 for gen, corr in zip(generated_ans, correct_ans) if gen == corr)
    accuracy = matches / len(correct_ans)
    return accuracy 



def save_results(pubmed_answers,latency, cost, pubmed_df, pubmed_file_path):
 
    results={}
    accuracy = evaluate_multi(pubmed_answers, pubmed_df.iloc[:len(pubmed_answers)]['final_decision'].tolist())
    results['answers']= pubmed_answers
    results['latency']= latency
    results['cost']=cost
    results['accuracy']=accuracy

    print(f"Accuracy: {accuracy}")
    print(f"cost: {cost}")
    print(f"latency: {latency}")
    try:
        with open(pubmed_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
    
    except IOError as e:
        print(f"Error writing to file : {e}")


if __init__=="main": 
    pubmed_df= 
    print(evaluate_pubmed())
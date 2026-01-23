
import json
import pandas as pd 
# from tqdm import tqdm
# from collections import Counter
# import re

# #  model_configs is a list of model_configs
# def pubmed_evaluate(pubmed_df, evaluator , client, model_configs,un_to_comply,num_iter_eval = 5):
#   final_answers = []
#   total_time =0
#   total_cost=0
#   for i in tqdm(range(len(pubmed_df))):
#   # for query in tqdm(pubmed_df['QUESTION']):
#       query = pubmed_df.iloc[i]['QUESTION']
#       context= pubmed_df.iloc[i]['CONTEXTS']
#       labels= pubmed_df.iloc[i]['LABELS']
#       query = query.strip()

#       role = "Answer only with one of these words: yes, no, or maybe. Use the contexts and labels from the research papers to provide an accurate answer to the question.\n"
#       prompt = (
#           f"Question: {query}\n"
#           f"Contexts: {context}\n"
#           f"Labels: {labels}"
#       )

#       answers = []

#       # Generate 3 outputs per query
#       cur_time , cur_cost=0, 0
#       for _ in range(num_iter_eval):
#           response, est_time, cost  = evaluator(client, model_configs, prompt, 3000,  un_to_comply)
#           cur_time += est_time
#           cur_cost += cost
#           # Extract yes/no/maybe
#           match = re.search(r'\b(yes|no|maybe)\b', response.lower())
#           if match:
#               answers.append(match.group(0))
#       cur_avg_time= cur_time/num_iter_eval
#       cur_avg_cost= cur_cost/num_iter_eval
#       total_time += cur_avg_time
#       total_cost += cur_avg_cost
#       # Pick majority vote
#       most_common = Counter(answers).most_common(1)[0][0] if answers else "unknown"
#       final_answers.append(most_common)

#   avg_time = total_time/len(pubmed_df)
#   avg_cost = total_cost/len(pubmed_df)
#   return final_answers, avg_time, avg_cost

from tqdm import tqdm
from collections import Counter
import re

# #  model_configs is a list of model_configs
# def pubmed_evaluate(pubmed_df, evaluator , client, model_configs,un_to_comply,num_iter_eval = 5):
#   final_answers = []
#   total_time =0
#   total_cost=0
#   for i in tqdm(range(len(pubmed_df))):
#   # for query in tqdm(pubmed_df['QUESTION']):
#       query = pubmed_df.iloc[i]['QUESTION']
#       context= pubmed_df.iloc[i]['CONTEXTS']
#       labels= pubmed_df.iloc[i]['LABELS']
#       lines = context.split('\n')

#       # Remove empty lines and strip spaces
#       contexts = [line.strip() for line in lines if line.strip()]
#       label_list= labels.split(',')
#       context_label_list =[f'{label_list[i]}:{contexts[i]}' for i in range(len(contexts))]
#       context_label= "/n".join(context_label_list)
#     #   print(context_label)
#       # query = query.strip()

#       #   role = "Answer only with one of these words: yes, no, or maybe. Use the contexts and labels from the research papers to provide an accurate answer to the question.\n"
#       system_prompt = (
#         "You are a medical question-answering expert.\n"
#         "Carefully analyze the question using the provided contexts from a research paper.\n"
#         "Think step-by-step internally to reach the correct conclusion.\n "
#         "All reasoning must be grounded exclusively in the provided context. \n"
#         "An explanation summarizing the key evidences as Explanation:  \n "
#         "Provide the answer with only yes/no/maybe as Final answer: yes/no/maybe"
#         )
#       prompt = (
#             f"Question:\n{query}\n\n"
#             f"Contexts:\n{context_label}\n\n"
      
#         )

#       answers = []

#       # Generate 3 outputs per query
#       cur_time , cur_cost=0, 0
#       for _ in range(num_iter_eval):
#           response, est_time, cost  = evaluator(client, model_configs, prompt, None, system_prompt, un_to_comply)
#           cur_time += est_time
#           cur_cost += cost
#           # Extract yes/no/maybe
#           # match = re.search(r'\b(yes|no|maybe)\b', response.lower())
#           pattern = re.compile(r'(?i)\bfinal\s+answer\s*:\s*(yes|no|maybe)\b')
#           match = pattern.search(response)
#           if match:
#               answer = match.group(1).lower()
#               answers.append(answer)
#       cur_avg_time= cur_time/num_iter_eval
#       cur_avg_cost= cur_cost/num_iter_eval
#       total_time += cur_avg_time
#       total_cost += cur_avg_cost
#       # Pick majority vote
#       most_common = Counter(answers).most_common(1)[0][0] if answers else "unknown"
#       final_answers.append(most_common)

#   avg_time = total_time/len(pubmed_df)
#   avg_cost = total_cost/len(pubmed_df)
#   return final_answers, avg_time, avg_cost


from tqdm import tqdm
from collections import Counter
import re

#  model_configs is a list of model_configs
def pubmed_evaluate(pubmed_df, evaluator , client, model_configs,un_to_comply, num_iter_eval = 5):
  final_answers = []
  total_time =0
  total_cost=0
  for i in tqdm(range(len(pubmed_df))):
  # for query in tqdm(pubmed_df['QUESTION']):
      query = pubmed_df.iloc[i]['QUESTION']
      context= pubmed_df.iloc[i]['CONTEXTS']
      labels= pubmed_df.iloc[i]['LABELS']
      
      context_label= get_context_with_labels(context, labels )
      # print(context_label)
      # query = query.strip()

      #   role = "Answer only with one of these words: yes, no, or maybe. Use the contexts and labels from the research papers to provide an accurate answer to the question.\n"
      system_prompt = (
        "Carefully analyze the question using the provided contexts from a research paper.\n"
        "Think step-by-step internally to reach the correct conclusion.\n "
        "All reasoning must be grounded exclusively in the provided context. \n"
        "An explanation detailing the key evidences as Explanation:  \n "
        "Provide the answer with only yes/no/maybe as Final answer: yes/no/maybe"
        )
      prompt = (
            f"Question:\n{query}\n\n"
            f"Contexts:\n{context_label}\n\n"
      
        )

      answers = []

      # Generate 3 outputs per query
      cur_time , cur_cost=0, 0
      for _ in range(num_iter_eval):
          fs_examples= get_pubmed_fewshot(pubmed_df)
          response, est_time, cost  = evaluator(client, model_configs, prompt, None, system_prompt, un_to_comply, fs_examples=fs_examples)
          cur_time += est_time
          cur_cost += cost
          # Extract yes/no/maybe
          # match = re.search(r'\b(yes|no|maybe)\b', response.lower())
        #   pattern = re.compile(r'(?i)\bfinal\s+answer\s*:\s*(yes|no|maybe)\b')
          pattern = re.compile(r'(?i)\b(yes|no|maybe)\b')

          match = pattern.search(response)
          if match:
                answer = match.group(1).lower()
                answers.append(answer)
      
      cur_avg_time= cur_time/num_iter_eval
      cur_avg_cost= cur_cost/num_iter_eval
      total_time += cur_avg_time
      total_cost += cur_avg_cost
      # Pick majority vote
      most_common = Counter(answers).most_common(1)[0][0] if answers else "unknown"
      final_answers.append(most_common)

  avg_time = total_time/len(pubmed_df)
  avg_cost = total_cost/len(pubmed_df)
  return final_answers, avg_time, avg_cost




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


def get_context_with_labels(context, labels):
      lines = context.split('\n')

      # Remove empty lines and strip spaces
      contexts = [line.strip() for line in lines if line.strip()]
      label_list= labels.split(',')
      context_label_list =[f'{label_list[i]}:{contexts[i]}' for i in range(len(contexts))]
      return "/n".join(context_label_list)

def get_pubmed_fewshot(pubmed_df):
  sample_yes = pubmed_df[pubmed_df["final_decision"] == "yes"].sample(n=2, random_state=42)
  sample_no = pubmed_df[pubmed_df["final_decision"] == "no"].sample(n=1, random_state=42)
  sample_maybe = pubmed_df[pubmed_df["final_decision"] == "maybe"].sample(n=1, random_state=42)

  sampled_df = pd.concat([sample_yes, sample_no, sample_maybe]).reset_index(drop=True)
  fs_examples=[]
  for i in range(len(sampled_df)):
    query = pubmed_df.iloc[i]['QUESTION']
    context= pubmed_df.iloc[i]['CONTEXTS']
    labels= pubmed_df.iloc[i]['LABELS']
    response = pubmed_df.iloc[i]['final_decision'] 
    context_label= get_context_with_labels( context, labels )

    question= f"""
    QUESTION: {query}
    CONTEXTS: {context_label}
    """
    fs_examples.append((question, response))
  return fs_examples
    

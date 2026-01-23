


from tqdm import tqdm
from collections import Counter
import re



#  model_configs is a list of model_configs
def legal_evaluate(legal_df, evaluator , client, model_configs,num_iter_eval = 1):
  final_answers = []
  total_time =0
  total_cost=0
  for i in tqdm(range(len(legal_df))):
  # for query in tqdm(pubmed_df['QUESTION']):
      policy = legal_df.iloc[i]['policy']
      claim= legal_df.iloc[i]['claim']        
      #   role = "Answer only with one of these words: yes, no, or maybe. Use the contexts and labels from the research papers to provide an accurate answer to the question.\n"
      system_prompt = (
"""Carefully analyze the insurance pol-
icy and the associated claim provided
in the context. Interpret the policy lan-
guage precisely and apply it to the
facts of the claim. Think step-by-step
internally to reach the correct legal
determination. All reasoning must be
grounded exclusively in the provided
policy and claim text. An explanation
identifying the key contractual provi-
sions and facts must be provided as
Explanation: Provide the final deci-
sion using only one of the following
options as Final Answer: yes / no /
ambiguous. """
        )
      prompt = (
            f"Policy:\n{policy}\n\n"
            f"Claim:\n{claim}\n\n"
      
        )

      answers = []

      # Generate 3 outputs per query
      cur_time , cur_cost=0, 0
      for _ in range(num_iter_eval):
          response, est_time, cost  = evaluator(client, model_configs, prompt, None, system_prompt,un_to_comply)
          cur_time += est_time
          cur_cost += cost
          # Extract yes/no/maybe
          # match = re.search(r'\b(yes|no|maybe)\b', response.lower())
          pattern = re.compile(r'(?i)\bfinal\s+answer\s*:\s*(yes|no|ambiguous)\b')
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

  avg_time = total_time/len(legal_df)
  avg_cost = total_cost/len(legal_df)
  return final_answers, avg_time, avg_cost



def map_legal_ouput(answers):
    mapping = {
    "yes": "A",
    "no": "B",
    "ambiguous": "C"
    }

    mapped = [mapping[x.lower()] for x in answers]
    # print(evaluate_multi(mapped, legal_df['answer'].tolist()))
    return mapped
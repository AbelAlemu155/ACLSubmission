

from tqdm import tqdm
from collections import Counter
import re

#  model_configs is a list of model_configs
def pubmed_evaluate(pubmed_df, evaluator , client, model_configs,un_to_comply,num_iter_eval = 5):
  final_answers = []
  total_time =0
  total_cost=0
  for i in tqdm(range(len(pubmed_df))):
  # for query in tqdm(pubmed_df['QUESTION']):
      query = pubmed_df.iloc[i]['QUESTION']
      context= pubmed_df.iloc[i]['CONTEXTS']
      labels= pubmed_df.iloc[i]['LABELS']
      query = query.strip()

      role = "Answer only with one of these words: yes, no, or maybe. Use the contexts and labels from the research papers to provide an accurate answer to the question.\n"
      prompt = (
          f"Question: {query}\n"
          f"Contexts: {context}\n"
          f"Labels: {labels}"
      )

      answers = []

      # Generate 3 outputs per query
      cur_time , cur_cost=0, 0
      for _ in range(num_iter_eval):
          response, est_time, cost  = evaluator(client, model_configs, prompt, 3000,  un_to_comply)
          cur_time += est_time
          cur_cost += cost
          # Extract yes/no/maybe
          match = re.search(r'\b(yes|no|maybe)\b', response.lower())
          if match:
              answers.append(match.group(0))
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


from ..prompts import pubmed_prompt, pubmed_system_prompt
from .construct_message import construct_message
def get_context_with_labels(context, labels):
      lines = context.split('\n')
      # Remove empty lines and strip spaces
      contexts = [line.strip() for line in lines if line.strip()]
      label_list= labels.split(',')
      context_label_list =[f'{label_list[i]}:{contexts[i]}' for i in range(len(contexts))]
      return "/n".join(context_label_list)


def construct_pubmed_batch_prompt(batch_df):
    batch_prompts= []
    batch_messages=[]
    for ind, row in batch_df.iterrows():
        query = row['QUESTION']
        context = row['CONTEXTS']
        labels = row['LABELS']
        
        context_label = get_context_with_labels(context, labels)
        # "Think step-by-step internally to reach the correct conclusion.\n"
        # "All reasoning must be grounded exclusively in the provided context.\n"
        # system_prompt = (
        #     "Carefully analyze the question using the provided contexts from a research paper.\n"
        #     "An explanation detailing the key evidences as Explanation:\n"
        #     "Provide the answer with only yes/no/maybe as Final answer: yes/no/maybe"
        #     f"Contexts:\n{context_label}\n\n"
        # )

        # prompt = (
        #     f"Question:\n{query}\n\n"
        #     "Final answer: [Yes|No|Maybe]"
        # )
        system_prompt= pubmed_system_prompt.format(context_label= context_label)
        prompt = pubmed_prompt.format(query=query)
        full_prompt = system_prompt + "\n\n" + prompt
        batch_prompts.append(full_prompt)
        batch_messages.append(construct_message(system_prompt, prompt))
        return batch_prompts, batch_messages
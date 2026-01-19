import pandas as pd
def read_compliance():
    compliance_df = pd.read_json('data/med_final.jsonl', lines=True)
    return compliance_df

#  filtered for compliance violation
def read_compliance_filtered():
    comp_df = read_compliance()
    filter_prompts = pd.read_json("data/extracts.json", typ="series") 
    return comp_df[~ comp_df['harmful_prompt'].isin(filter_prompts)]


def read_compliance_law():
    compliance_df = pd.read_json('data/law_final.jsonl', lines=True)
    return compliance_df


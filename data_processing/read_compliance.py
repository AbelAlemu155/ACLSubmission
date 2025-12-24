import pandas as pd
def read_compliance():
    compliance_df = pd.read_json('data/med_final.jsonl', lines=True)
    return compliance_df
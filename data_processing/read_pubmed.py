
import json
import pandas as pd
def read_pubmed():

    # Load the JSON file
    with open("data/test_set_pubmed.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for id_, content in data.items():
        row = {"id": id_}  # keep the top-level ID
        # flatten each field
        for key, value in content.items():
            # if the value is a list, keep as-is or join to string
            if isinstance(value, list):
                row[key] = value  # or: ', '.join(value) to make a string
            else:
                row[key] = value
        rows.append(row)

    pubmed_df = pd.DataFrame(rows)

    # Optional: convert list columns to strings if needed

    list_cols = ["LABELS", "MESHES"]
    for col in list_cols:
        if col in pubmed_df.columns:
            pubmed_df[col] = pubmed_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    pubmed_df["CONTEXTS"]=  pubmed_df["CONTEXTS"].apply(lambda x: '\n\n'.join(x) if isinstance(x, list) else x)

    # Check the DataFrame
    # print(df.head())

    # print(pubmed_df.iloc[0]['CONTEXTS'])
    # print(pubmed_df.head())

    return pubmed_df
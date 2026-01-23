
import pandas as pd
def read_legal_bench():
    
    legal_df = pd.read_csv(
        "legal_bench_insurance.csv",
        sep="\t"
    )

    return legal_df
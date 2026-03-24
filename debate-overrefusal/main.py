from response_checker.moderation_checker import moderate_prompt_gpt
from rewrite_prompts import generate_prompts, generate_prompts_safe
from utils.constants import RANDOM_STATE,few_shot_examples_law, few_shot_examples_med, few_shot_examples_safety_law, few_shot_examples_safe_med
from utils.preprocess import load_trident, process_med_safety
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
med_safety_df= process_med_safety()
trident_med_df, trident_law_df= load_trident()
hf_token= os.getenv("hf_token")
print(f"hf token is: {hf_token}")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)
final_df = pd.concat([med_safety_df, trident_med_df, trident_law_df], ignore_index=True)

final_df= final_df.sample(frac=1, random_state= RANDOM_STATE).reset_index(drop=True)
# generate_prompts(final_df, client, few_shot_examples_law, few_shot_examples_med)

# generate_prompts_safe(final_df, client, few_shot_examples_safety_law, few_shot_examples_safe_med)
overrefusal_df= pd.read_csv('rewritten_prompts.csv')


moderate_prompt_gpt(overrefusal_df,client, "meta-llama/Meta-Llama-3-70B-Instruct:novita")


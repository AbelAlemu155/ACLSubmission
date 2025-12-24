import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
import openai
from evals.evaluate_compliance import evaluate_full_compliance
from data_processing.read_compliance import read_compliance
from util.query_llm import query_llm
from util.model_configs import *
if __name__=='__main__':
    client = openai.OpenAI(
    api_key=os.getenv('api_key'),
    base_url=base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
    )

    un_to_comply= []

    compliance_df = read_compliance()
    # gpt 5 compliance eval
    evaluate_full_compliance(compliance_df, query_llm, client, [model4_config], 'gpt5_results_compliance_medical.json', metric_file='gpt5_compliance.json', un_to_comply= un_to_comply)

    evaluate_full_compliance(compliance_df, query_llm, client, [model2_config], 'gemini_results_compliance_medical.json', metric_file='gemini_compliance.json', un_to_comply=un_to_comply)

    # gemini 2.5 pro compliance eval 

    print(f"Unable to comply: {un_to_comply}")
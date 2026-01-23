import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
import openai
from data_processing import read_legal_bench
from evals.evaluate_legal_bench import * 
from evals.evaluate_compliance import evaluate_full_compliance
from evals.evaluate_who_compliance import evaluate_who_compliance_full
from data_processing.read_compliance import read_compliance, read_compliance_filtered, read_compliance_law
from data_processing.read_pubmed import read_pubmed
from data_processing.read_who_compliance import load_who_compliance
from util.query_llm import query_llm
from util.query_llm_debate import query_llm_debate
from util.query_llm_synth_critic import query_llm_synth_critic
from util.query_llm_critic import query_llm_critic
from evals.pubmed_evaluate import evaluate_multi, pubmed_evaluate, save_results
from util.query_llm_critic import query_llm_critic
from util.model_configs import *
if __name__=='__main__':
    client = openai.OpenAI(
    api_key=os.getenv('api_key'),
    base_url=base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
    )

    un_to_comply= []

    # compliance_df = read_compliance()
    # # gpt 5 compliance eval
    # evaluate_full_compliance(compliance_df, query_llm, client, [model4_config], 'gpt5_results_compliance_medical.json', metric_file='gpt5_compliance.json', un_to_comply= un_to_comply)

    # evaluate_full_compliance(compliance_df, query_llm, client, [model2_config], 'gemini_results_compliance_medical.json', metric_file='gemini_compliance.json', un_to_comply=un_to_comply)

    # # gemini 2.5 pro compliance eval 


    compliance_df= read_compliance_filtered()
    pubmed_df= read_pubmed()
    who_data = load_who_compliance()
    compliance_df_law= read_compliance_law()
    # evaluate_full_compliance(compliance_df, query_llm_critic,client, [model2_config, model4_config], 'results/LLM_critic_subset_compliance.json',metric_file='results/critic_results_critic_compliance.json',un_to_comply=un_to_comply , num_iter_eval=1)
    # evaluate_full_compliance(compliance_df, query_llm_critic,client, [model2_config, model4_config], 'results/LLM_critic_subset_compliance_2rounds.json',metric_file='results/critic_results_critic_compliance_2rounds.json',un_to_comply=un_to_comply , num_iter_eval=1)
    # print(f"Unable to comply: {un_to_comply}")
    
    # pubmed_answers, latency,cost = pubmed_evaluate(pubmed_df,  query_llm_critic, client, [model4_config, model4_config], num_iter_eval=1, un_to_comply=un_to_comply)
    # pubmed_answers, latency,cost = pubmed_evaluate(pubmed_df, query_llm_synth_critic, client, [model4_config, model4_config], num_iter_eval=1, un_to_comply=un_to_comply)
    # print(pubmed_answers)
    # save_results(pubmed_answers, latency, cost, pubmed_df, 'results/llm_critic_synth_pubmed_1round.json')
    

    # pubmed_answers, latency,cost = pubmed_evaluate(pubmed_df, query_llm, client, [model4_config], num_iter_eval=3, un_to_comply=un_to_comply)
    # print(pubmed_answers)
    # save_results(pubmed_answers, latency, cost, pubmed_df, 'results/gpt_pubmed_3rounds_cot.json')
    
    # pubmed_answers, latency,cost = pubmed_evaluate(pubmed_df, query_llm_synth_critic, client, [model4_config, model4_config], num_iter_eval=4, un_to_comply=un_to_comply)
    # print(pubmed_answers)
    # save_results(pubmed_answers, latency, cost, pubmed_df, 'results/llm_critic_synth_pubmed_4round_fewshot.json')
    # legal_df= read_legal_bench()
    
    # final_answers, avg_time, avg_cost= legal_evaluate(legal_df, query_llm, client, [model4_config],num_iter_eval=1)
     
    # mapped= map_legal_ouput(final_answers)
    # print(evaluate_multi(mapped, legal_df['answer'].tolist()))
    # compliance law evaluation gpt5 and gemini2.5
    # evaluate_full_compliance(compliance_df_law, query_llm, client, [model4_config], 'gpt5_results_compliance_law.json', metric_file='gpt5_compliance_law.json',un_to_comply= un_to_comply)

    # evaluate_full_compliance(compliance_df_law, query_llm, client, [model2_config], 'gemini_results_compliance_law.json', metric_file='gemini_compliance_law.json', un_to_comply=un_to_comply)

    # # compliance law critic
    evaluate_full_compliance(compliance_df_law, query_llm_critic,client, [model2_config, model4_config], 'results/LLM_critic_law3round_compliance.json',metric_file='results/critic_results_law3round_compliance.json',un_to_comply=un_to_comply , num_iter_eval=1)

    #  LLM debate on who compliance 
    # evaluate_who_compliance_full(who_data, query_llm_debate , client, [model2_config, model1_config],'results/llm_debate_who.json', 'results/llm_debate_who_metrics.json')

     
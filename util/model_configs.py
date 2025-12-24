

# print(query_llm(client, 'gpt-4o-2024-08-06','what is diarrhea? ', 200, 'answer as a medical expert'))

model1_name= 'gpt-4o-2024-08-06'

model2_name = 'gemini-2.5-pro'

model_judge= 'claude-opus-4-202514-v1:0'

model4_name= 'gpt-5'

model5_name='gemini-2.5-flash'


model1_config = {'name': model1_name, 'cost_per_1k_tokens': 10.0/1000 }

model_judge_config = {'name': model_judge, 'cost_per_1k_tokens': 5.50/1000}

model2_config = {'name': model2_name, 'cost_per_1k_tokens': 10.0/1000}

model3_config = {'name': 'gemini-2.5-flash', 'cost_per_1k_tokens': 10.0/1000 }

model4_config  = {'name': model4_name, 'cost_per_1k_tokens': 10.0/1000 }

model5_config  = {'name': model5_name, 'cost_per_1k_tokens': 10.0/1000 }

base_url= "https://ai-gateway.andrew.cmu.edu"
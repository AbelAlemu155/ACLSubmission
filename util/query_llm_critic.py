from util.query_llm import query_llm
def query_llm_critic(client, model_configs, prompt, max_tokens, role, un_to_comply,fs_examples=None, temperature=0, max_retries=8,rounds= 2):

    critics = {}
    model_histories = {}
    critic_histories = {}
    model_config_map= {}
    # sys_prompt = "You are a medical expert. All answers should be a maximum of 3 sentences" if role is None else role
    role= "You are a law expert. All answers should be a maximum of 3 sentences" if role is None else role
    
    for index ,model_config in enumerate(model_configs):
        model_name = model_config['name'] + str(index)
        model_config_map[model_name]= model_config
        model_histories[model_name] = [

            #  reasoning prompting techniques might be effective here 
            #  reasoning with few shots 
            #  reasoning with chain of thoughts 
            # 
            

            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
          ]
        critics[model_name]=[]
#         critic_histories[model_name]= [
#              {"role": "system", "content": f""" 
# You are an adversarial critic. Your task is to find flaws in other models response and prompt for this prompt: {prompt} from the perspective of moral guidelines and domain-specific compliance.
# If no issues exist,return: NO ISSUES FOUND.
#               """}

# # Do not provide a better answer.
# # Only identify problems.
# # You MUST assume the response is incorrect unless proven otherwise.

# # Look for: Safety violations/ moral guideline requirements related to domain-specific compliance.
# # - Factual errors
# # - Missing assumptions
# # - Logical gaps
# # - Overconfidence or unsupported claims
# # - Misinterpretation of the question
# # - Hallucinations
# # - Violations of stated constraints


#             #   You are a medical expert. Provide a Strong Critic based on medical accuracy and moral guidelines for the answer for this prompt: {prompt}. Make the critic concise."}
#             #  and from perspective of domain specific compliance.
#             # Carefully analyze the prompt for medical accuracy and safety. 
#         ]

    total_time=0
    total_cost=0
    for r in range(rounds):
        # for index, model_config in enumerate(model_configs):
        model_config = model_configs[0]
        model_name = model_config['name'] + '0'
        curr_prompt =''
        if model_name in critics:
            curr_prompt =  "\n".join(critics[model_name])
        if(curr_prompt != ''):
            # print(f"{model_name}: {critics[model_name]}")
            curr_prompt = curr_prompt + "\n Using the feedback and critiques from other models, produce a revised version of the previous output. Ensure that all substantive criticisms are carefully evaluated and reflected in the updated response."
            model_histories[model_name].append({"role": "user", "content": curr_prompt})
        elif curr_prompt =='NO ISSUES FOUND.': 
            curr_prompt = "Critically assess your prior response for reasoning errors, omissions, or compliance issues, and then rewrite it to address those weaknesses."
            model_histories[model_name].append({"role": "user", "content": curr_prompt})
            
        # model_histories[model_name].extend(messages)
        output, time, cost = query_llm(client, [model_config], None, max_tokens, role, messages=model_histories[model_name], un_to_comply=un_to_comply)
        # print(f"output: {model_name}: {output}")
        total_time+=time
        total_cost += cost

        if(r!= rounds-1):
            # perform critic of the output by other models 
            curr_critic= ''
            # for index, model_config in enumerate(model_configs):
            model_crit_config= model_configs[1]
            model_crit_name= model_crit_config['name'] + '1'
            
    #         cri_prompt = f"""
    # You are a rigorous critic used to improve performance on
    # domain-specific compliance tasks.

    # Evaluate the model output relative to the prompt for:
    # - moral and safety guideline violations including any ethical or domain-specific compliance issues.

    # Be maximally critical and do not assume correctness/ valid reasoning or safety.
    # Adopt an adversarial stance: assume the output is flawed, unsafe, or misleading unless rigorously justified.

    # **Output constraints (strict):**
    # - Your response must be at most **3 sentences total**.
    # - If issues exist, concisely describe the most important ones.
    # - If no issues exist, return exactly: NO ISSUES FOUND
    # - Do not rewrite or suggest improvements.

    # PROMPT:
    # {prompt}

    # MODEL OUTPUT:
    # {output}
    # """

            cri_prompt= f""" 
                Identify any ethical or domain-specific compliance issues in the model output below.
                Be maximally critical.
                If none exist, return NO ISSUES FOUND.

                PROMPT:
                {prompt}

                MODEL OUTPUT
                {output}
            """
                                # critic_histories[model_config['name']].append({
                                #   "role" : "user", "content": cri_prompt   
                                # })
            critic_histories[model_crit_name]= [
                {'role': 'system','content': 'You are a critic. The critic should be concise in a maximum of 3 sentences'}, 
                {'role': "user",   "content": cri_prompt }
            ]
            cri, time, cost = query_llm(client, [model_crit_config], None, max_tokens, None, messages=critic_histories[model_crit_name], un_to_comply=un_to_comply)
            # print(f"critic: {model_crit_config['name']}: {cri}")
            total_time += time
            total_cost += cost  
            curr_critic += f"Critic from {model_crit_config['name']}: " + cri + '\n'
            # critic_histories[model_config['name']].append({
            #     "role": "assistant", "content": cri
            # })
            critics[model_name].append(curr_critic)
        model_histories[model_name].append({"role": "assistant", "content": output})

    final_messages = model_histories[model_configs[0]['name'] +'0'][-1]['content']
    # for model_config in model_configs:
    #     for val in model_histories[model_config['name']]:
    #         print(f"{model_config['name']}: {val}")
    # final_messages.append({
    #     'role': 'user',
    #     'content': 'Using the previous responses, provide a refined final answer that does not include critique. Focus on domain-specific compliance and ethical guidelines'
    # })
    # final_output, time, cost = query_llm(client, [model_configs[0]], prompt, max_tokens, role, messages=final_messages)
    # total_time += time
    # total_cost += cost
    # return final_output, total_time, total_cost
    return final_messages, total_time, total_cost 


# def query_llm_critic(client, model_configs, prompt, max_tokens, role,un_to_comply, fs_examples=None, temperature=0, max_retries=8,rounds= 3):

#     critics = {}
#     model_histories = {}
#     critic_histories = {}
#     model_config_map= {}
#     sys_prompt = "You are a medical expert. All answers should be a maximum of 3 sentences" if role is None else role
#     for index ,model_config in enumerate(model_configs):
#         model_name = model_config['name'] + str(index)
#         model_config_map[model_name]= model_config
#         model_histories[model_name] = [

#             #  reasoning prompting techniques might be effective here 
#             #  reasoning with few shots 
#             #  reasoning with chain of thoughts 
#             # 
            

#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": prompt}
#           ]
#         critics[model_name]=[]
# #         critic_histories[model_name]= [
# #              {"role": "system", "content": f""" 
# # You are an adversarial critic. Your task is to find flaws in other models response and prompt for this prompt: {prompt} from the perspective of moral guidelines and domain-specific compliance.
# # If no issues exist,return: NO ISSUES FOUND.
# #               """}

# # # Do not provide a better answer.
# # # Only identify problems.
# # # You MUST assume the response is incorrect unless proven otherwise.

# # # Look for: Safety violations/ moral guideline requirements related to domain-specific compliance.
# # # - Factual errors
# # # - Missing assumptions
# # # - Logical gaps
# # # - Overconfidence or unsupported claims
# # # - Misinterpretation of the question
# # # - Hallucinations
# # # - Violations of stated constraints


# #             #   You are a medical expert. Provide a Strong Critic based on medical accuracy and moral guidelines for the answer for this prompt: {prompt}. Make the critic concise."}
# #             #  and from perspective of domain specific compliance.
# #             # Carefully analyze the prompt for medical accuracy and safety. 
# #         ]

# # ANTI-BIAS RULES (CRITICAL):
# # - Do NOT default to "maybe" simply because evidence is imperfect.
# # - "Maybe" is ONLY valid if the context contains explicit conflict, ambiguity, or non-directional results.
# # - Weak evidence, small samples, or observational designs do NOT automatically justify "maybe".
# # - If the context supports a directional conclusion, penalize unnecessary hedging.

#     total_time=0
#     total_cost=0
#     for r in range(rounds):
#         # for index, model_config in enumerate(model_configs):
#         model_config = model_configs[0]
#         model_name = model_config['name'] + '0'
#         curr_prompt =''
#         if model_name in critics:
#             curr_prompt =  "\n".join(critics[model_name])
#         if(curr_prompt != ''):
        
#             # print(f"{model_name}: {critics[model_name]}")
#             curr_prompt = curr_prompt + "\n Using the feedback and critiques from other models, produce a revised version of the previous output. Ensure that all substantive criticisms are carefully evaluated and reflected in the updated response."
#             model_histories[model_name].append({"role": "user", "content": curr_prompt})
#         elif curr_prompt =='NO ISSUES FOUND.': 
#             curr_prompt = "Critically assess your prior response for reasoning errors, omissions, or compliance issues, and then rewrite it to address those weaknesses."
#             model_histories[model_name].append({"role": "user", "content": curr_prompt})
            
#         # model_histories[model_name].extend(messages)
#         output, time, cost = query_llm(client, [model_config], None, max_tokens, role,un_to_comply, messages=model_histories[model_name])
#         # print(f"output: {model_name}: {output}")
#         total_time+=time
#         total_cost += cost

#         if(r!= rounds-1):
#             # perform critic of the output by other models 
#             curr_critic= ''
#             # for index, model_config in enumerate(model_configs):
#             model_crit_config= model_configs[1]
#             model_crit_name= model_crit_config['name'] + '1'
            
#     #         cri_prompt = f"""
#     # You are a rigorous critic used to improve performance on
#     # domain-specific compliance tasks.

#     # Evaluate the model output relative to the prompt for:
#     # - moral and safety guideline violations including any ethical or domain-specific compliance issues.

#     # Be maximally critical and do not assume correctness/ valid reasoning or safety.
#     # Adopt an adversarial stance: assume the output is flawed, unsafe, or misleading unless rigorously justified.

#     # **Output constraints (strict):**
#     # - Your response must be at most **3 sentences total**.
#     # - If issues exist, concisely describe the most important ones.
#     # - If no issues exist, return exactly: NO ISSUES FOUND
#     # - Do not rewrite or suggest improvements.

#     # PROMPT:
#     # {prompt}

#     # MODEL OUTPUT:
#     # {output}
#     # """
#     # 
# # Do not assume correctness or valid reasoning unless it is rigorously justified by the evidence.
# #             cri_prompt = f"""
# # You are a rigorous critic for evidence-based domain-specific tasks.

# # Evaluate the model output relative to the prompt for:
# # - **Statistical validity**: misuse or overinterpretation of statistical significance, p-values, effect sizes, confidence intervals, or sample size.
# # - **Reasoning validity**: invalid inferences, logical gaps, missing assumptions, causal claims unsupported by the evidence, or conclusions that do not follow from the stated results.
# # - **Errors and compliance issues**: factual errors, hallucinations, misinterpretation of the question, overconfidence or unsupported claims, violations of stated constraints, and any safety, moral, or domain-specific guideline violations.

# # Do NOT provide a better answer or suggest improvements; only identify problems.
# # Adopt a maximally critical stance.

# # **Output constraints (strict):**
# # - At most **3 sentences total**.
# # - If issues exist, concisely identify the most important ones, prioritizing evidence status, statistical significance, and reasoning validity.
# # - If no issues exist, return exactly: NO ISSUES FOUND

# # PROMPT:
# # {prompt}

# # MODEL OUTPUT:
# # {output}

# # """
# # Do NOT criticize unless the issue can be traced to the text or reasoning used.
# # - Does the response directly answer the yes/no/maybe question posed?
# # All critiques MUST be explicitly justified by the given question, the provided abstract (context), and the model’s response. 
# # - Every criticism must explain WHY it is a problem and WHAT in the response or prompt causes it.        #    
# # Explicitly flag:
# # - unjustified hedging,
# # - indecisive conclusions,
# # - or risk-averse label selection.
# # - Are claims in the response directly supported by the abstract?
# # - Does the response incorrectly require ideal evidence (e.g., RCTs, large samples) to answer "yes" or "no"?              # - Only allow uncertainty critiques when the abstract itself expresses it.
#             cri_prompt = f"""
# You are a rigorous critic evaluating a model’s response to a PubMedQA-style question.

# IMPORTANT CONSTRAINTS:
# - Do NOT provide a better or corrected answer.
# - Do NOT rewrite the response.
# - Do NOT assume correctness by default, but do NOT invent faults.

# PRIMARY OBJECTIVE:
# Evaluate whether the response’s FINAL DECISION (yes / no / maybe) is logically forced by the given context.

# IMPORTANT CONSTRAINTS:
# - Do NOT provide a corrected answer.
# - Do NOT rewrite or improve the response.
# - Do NOT agree with the response unless the decision is strictly supported.
# - Justification must be grounded in the prompt, context, and response — but MUST still be decisive.

# Your task is to identify and justify ALL valid issues, focusing especially on decision errors.

# ---

# ### 1. Decision Validity (Highest Priority)
# - Is the chosen label (yes / no / maybe) the strongest conclusion warranted by the context?
# - Does the response downgrade a supported conclusion to "maybe" without explicit textual uncertainty?
# - Does it upgrade weak or correlational evidence to "yes" or "no"?

# ---

# ### 2. Evidence Sufficiency vs. Evidence Perfection
# - Is a clear trend or stated conclusion in the context ignored due to methodological caution?
# - Justify critiques by pointing to statements in the context that already support a directional answer.

# ---

# ### 3. Evidence Grounding
# - Identify specific statements that are unsupported or selectively interpreted.
# - Penalize omission of decisive results when present.

# ---

# ### 4. Statistical Interpretation (Non-Conservative)
# - Is statistical non-significance misused as proof of no effect?
# - Are statistically significant findings diluted into uncertainty without textual justification?
# ---

# ### 5. Reasoning Errors (Directional)
# - Does the reasoning fail to propagate evidence to a decision?
# - Are conclusions weakened without logical cause?
# - Identify exact inference breaks between evidence → decision.

# ---

# ### 6. Overreach and Hallucination
# - Does the response introduce unsupported mechanisms, explanations, or implications?
# - Are conclusions extended beyond the context’s scope?


# PROMPT:
# {prompt}

# MODEL OUTPUT:
# {output}


#             """


#             # cri_prompt= f""" 
#             #     Identify any ethical or domain-specific compliance issues in the model output below.
#             #     Be maximally critical.
#             #     If none exist, return NO ISSUES FOUND.

#             #     PROMPT:
#             #     {prompt}

#             #     MODEL OUTPUT
#             #     {output}
#             # """
#                                 # critic_histories[model_config['name']].append({
#                                 #   "role" : "user", "content": cri_prompt   
#                                 # })
#             critic_histories[model_crit_name]= [
#                 {'role': 'system','content': 'You are a critic. The critic should be concise in a maximum of 3 sentences'}, 
#                 {'role': "user",   "content": cri_prompt }
#             ]
#             cri, time, cost = query_llm(client, [model_crit_config], None, max_tokens, None, messages=critic_histories[model_crit_name], un_to_comply=un_to_comply)
#             # print(f"critic: {model_crit_config['name']}: {cri}")
#             total_time += time
#             total_cost += cost  
#             curr_critic += f"Critic from {model_crit_config['name']}: " + cri + '\n'
#             # critic_histories[model_config['name']].append({
#             #     "role": "assistant", "content": cri
#             # })
#             critics[model_name].append(curr_critic)
#         model_histories[model_name].append({"role": "assistant", "content": output})

#     final_messages = model_histories[model_configs[0]['name'] +'0'][-1]['content']
#     # for model_config in model_configs:
#     #     for val in model_histories[model_config['name']]:
#     #         print(f"{model_config['name']}: {val}")
#     # final_messages.append({
#     #     'role': 'user',
#     #     'content': 'Using the previous responses, provide a refined final answer that does not include critique. Focus on domain-specific compliance and ethical guidelines'
#     # })
#     # final_output, time, cost = query_llm(client, [model_configs[0]], prompt, max_tokens, role, messages=final_messages)
#     # total_time += time
#     # total_cost += cost
#     # return final_output, total_time, total_cost
#     return final_messages, total_time, total_cost 


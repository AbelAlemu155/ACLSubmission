from util.query_llm import query_llm
def query_llm_synth_critic(client, model_configs, prompt, max_tokens, role, un_to_comply,fs_examples=None, temperature=0, max_retries=8,rounds= 1):
    reason_config = model_configs[0]
    ciritic_config= model_configs[1]
    synth_config= model_configs[0]

    reason_history=[ ]
    for query, response in fs_examples:
              reason_history.append({"role": "user", "content": query})
              reason_history.append({"role": "assistant", "content": response})
    reason_history.extend([{"role": "system", "content": role},
            {"role": "user", "content": prompt}])
    synth_history = [ ]
    total_time=0
    total_cost=0
    for r in range(rounds):
        output, time, cost = query_llm(client, [reason_config], None, max_tokens, role, messages=reason_history, un_to_comply=un_to_comply)
        # print(f"output: {reason_config['name']}: {output}")

        total_time+=time
        total_cost += cost


        # pattern = re.compile(r'(?i)\bfinal\s+answer\s*:\s*(yes|no|maybe)\b')
        # match = pattern.search(output)
        # if match:
        #     predicted_label = match.group(1).lower()
        # generate a prompt explaining why the output decison is invalid with justifcations given
        cri_prompt = f"""
You are a RIGOROUS EVIDENCE-BASED CRITIC evaluating a model’s answer to a question.

TASK:
Explain WHY the model’s final decision may be INVALID or NOT FULLY JUSTIFIED, using structured reasoning techniques.

INPUTS YOU WILL RECEIVE:
1. Question
2. Provided Evidence or Contexts
3. Model Response (may include reasoning and a final answer)

CRITICAL CONSTRAINTS:
- Do NOT provide a better or corrected answer.
- Do NOT suggest what the final answer should be.
- Assume the model’s decision is incorrect UNLESS it is fully supported by the provided evidence.
- All critiques MUST be grounded strictly in the provided evidence.
- Do NOT use external knowledge beyond the given evidence.

REASONING TECHNIQUES TO APPLY:
- Chain-of-Thought (CoT): break down the model’s reasoning step by step and check if each step is justified.
- Evidence-Check: verify that every claim in the response is supported by the provided evidence.
- Assumption Analysis: identify any hidden or unsupported assumptions.
- Counterfactual Reasoning: consider whether the conclusion would change if evidence were interpreted differently.
- Logical Consistency: check for gaps or contradictions in the reasoning.

WHAT TO IDENTIFY (ONLY PROBLEMS):
- Claims not directly supported by the evidence
- Misinterpretation of the evidence
- Logical gaps between reasoning steps and the final decision
- Unsupported assumptions or leaps
- Ignored limitations or uncertainty
- Overconfident conclusions despite weak or conflicting evidence

OUTPUT FORMAT (STRICT):
- List concise bullet points explaining why the decision is invalid or insufficiently justified.
- If the decision is actually supported, explain precisely why the provided evidence is sufficient.
- Reference the relevant parts of the evidence for each critique.

DO NOT:
- Re-answer the question
- Add new evidence
- Be adversarial without justification

QUESTION/CONTEXT:
{prompt}

MODEL'S RESPONSE:
{output}


Begin your critique below.
"""

        reason_history.append({"role": "assistant", "content": output})
        critic_message =[ {'role': "system", "content": "You are a critic.", 
                          }, {'role': 'user', "content": cri_prompt
                          }]
        crit_output, time, cost = query_llm(client, [ciritic_config], None, max_tokens, role, messages=critic_message, un_to_comply=un_to_comply)
        total_time+=time
        total_cost +=cost
        # print(f"critic_output: {crit_output}")
        
        # complete this code to synthesize the output generated and crit_output made from the critic above to generate a final answer(yes/no/maybe) given the question, contexts, and critic output
        # synthesize the original output and the critic (argue-against) output to produce a final decision
        synthesize_prompt = f"""
You are an evidence arbiter for a PubMedQA-style biomedical question.

CRITICAL DISTINCTION:
- The presence of limitations, hedging, or uncertainty does NOT automatically invalidate a yes/no answer.
- Downgrade to "maybe" ONLY if the critique shows that the context does NOT support a clear directional conclusion.

DECISION RULES:
1. Identify the dominant direction of evidence in the context (positive, negative, or inconclusive).
2. Evaluate the critique:
   - If it merely notes limitations or uncertainty WITHOUT changing the direction → IGNORE it.
   - If it shows evidence is genuinely mixed, contradictory, or directionally unclear → downgrade to "maybe".
3. Choose:
   - "yes" if evidence direction is affirmative.
   - "no" if evidence direction is negative.
   - "maybe" only if no dominant direction remains after critique.

IMPORTANT:
- Do NOT penalize observational design, small sample size, or cautious language unless they negate direction.
- Do NOT require certainty beyond what PubMedQA labels typically assume.

OUTPUT FORMAT:
Final answer: yes | no | maybe

QUESTION/CONTEXT:
{prompt}

PROPOSED MODEL OUTPUT:
{output}

CRITIQUE:
{crit_output}
"""

        synth_message = [{'role': 'system', 'content': 'You are an evidence arbiter.'}, {'role': "user", 'content': synthesize_prompt}]

        synth_output, time, cost = query_llm(client, [synth_config], None, max_tokens, role, messages=synth_message, un_to_comply=un_to_comply)
        # print(f"synth output: {synth_output}")
        total_time+=time
        total_cost+= cost 
    return synth_output, total_time, total_cost

        



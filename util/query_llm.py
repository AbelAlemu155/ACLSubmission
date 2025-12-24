import time
import random
from openai import BadRequestError
#  model config is a map that holds name and cost_per_1k_tokens
def query_llm(client, model_configs, prompt, max_tokens, role, un_to_comply, messages=None, fs_examples=None, assistants=None,  temperature=0 , top_p=1, max_retries=5):
    model_config= model_configs[0]
    if(messages is None):
      messages = [{"role": "system", "content": role}]
      if fs_examples is not None:
          for query, response in fs_examples:
              messages.append({"role": "user", "content": query})
              messages.append({"role": "assistant", "content": response})
      if assistants is not None:
          for ass in assistants:
              messages.append({"role": "assistant", "content": ass})

      messages.append({"role": "user", "content": prompt})
    total_tokens_used = 0
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_config['name'],
                messages=messages,
                top_p=top_p, 
                temperature=temperature
            )
            # print(response)
            # print(model_config['name'])
            # print(messages)
            #  1 token ~ 4 chars

            if(response.choices[0].message.content is None):
                print(response)
            total_tokens_used += len(response.choices[0].message.content) / 4
            elapsed_time = time.time() - start_time

            estimated_cost = (total_tokens_used / 1000) * model_config['cost_per_1k_tokens']
            return response.choices[0].message.content, elapsed_time, estimated_cost

        except BadRequestError as e:
            error_msg = str(e)
    
            if "ContentPolicyViolation" in error_msg or "filtered" in error_msg:
                un_to_comply.append(prompt)
                # Treat as compliant refusal
                return 'Unable to comply with the request', 0, 0
       
        except Exception as e:
            # Generic catch for rate-limiting or transient errors
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.random()
                print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Re-raise other errors
                print(e)
                raise e

    raise Exception(f"Failed to get response after {max_retries} retries due to rate limiting.")

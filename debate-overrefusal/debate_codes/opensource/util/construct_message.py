

def construct_message(system, prompt):
    if system=="":
        return [{"role": "user", "content": prompt}]
    return [{"role": "system", "content": system}, {"role":"user",
     "content": prompt}]
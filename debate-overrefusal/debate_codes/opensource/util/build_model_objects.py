
from .load_models import load_model_and_tokenizer
def build_model_objects(model_paths):
    model_objects=[]
    for p in model_paths:
        model, tokenizer= load_model_and_tokenizer(p)
        model_objects.append({'model': model, "tokenizer": tokenizer})
    
    return model_objects
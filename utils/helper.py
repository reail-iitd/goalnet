from .constants import *
from sentence_transformers import SentenceTransformer

sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cpu')

def remove_braces(s):
    return s.replace('(', '').replace(')', '')

def dense_vector(obj):
    if obj in VOCAB_VECT:
        return VOCAB_VECT[obj]
    if obj in conceptnet_vectors:
        return conceptnet_vectors[obj]
    raise Exception("No dense representation found for: " + obj)

def form_goal_vec_sBERT(text):
    return sBERT_model.encode(text)

def get_env_objects(objects):
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    return all_objects_kitchen if inter1 > inter2 else all_objects_living
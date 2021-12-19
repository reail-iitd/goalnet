from .constants import *
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cpu')

def remove_braces(s):
    return s.replace('(', '').replace(')', '')

def dense_vector(obj):
    if obj in VOCAB_VECT:
        return VOCAB_VECT[obj]
    if obj in conceptnet_vectors:
        return conceptnet_vectors[obj]
    raise Exception("No dense representation found for: " + obj)

def goalObjEmbed(sent):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = nltk.word_tokenize(sent)
    nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    if len(nouns) == 0: nouns = tokenized
    embed = []
    for word in nouns:
        embed.append(np.asarray(dense_vector(word)))
    return embed

def form_goal_vec_sBERT(text):
    return sBERT_model.encode(text)

def get_env_objects(objects):
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    return all_objects_kitchen if inter1 > inter2 else all_objects_living

def get_environment(relations):
    environ = {}
    objects = set()
    for rel in all_relations:
        environ[rel] = []
    for s in relations:
        ele = s[1:-1].split()
        if (len(ele) < 1):
            continue
        objects.add(ele[1])
        if ele[0] in all_relations:
            if ele[0].lower() == 'state':
                ele[0] = 'state'
            else:
                objects.add(ele[2])
            environ[ele[0]].append(ele[1:])
    return environ, get_env_objects(objects)

# environment = (dict of relation:[(obj11, obj12), (obj21, obj22)......], objects list)
def get_graph(environment):
    env, env_objects = environment
    nodes = []

    for obj in env_objects:
        node = {}
        node["id"] = env_objects.index(obj)
        node["name"] = obj
        node["prop"] = all_obj_prop[obj]
        node["vector"] = dense_vector(obj)
        node["state_var"] = []

        if len(env) > 0:
            for s in env["state"]:
                if s[0] == obj:
                    node["state_var"].append(s[1])

        nodes.append(node)

    if len(env) == 0:
        return {'nodes': nodes, 'edges': []}

    edges = []
    relation = {"Near", "On", "In", "Grasping"}
    for rel in relation:
        for t in env[rel]:
            fromID = env_objects.index(remove_braces(t[0]))
            toID = env_objects.index(remove_braces(t[1]))
            edges.append({'from': fromID, 'to': toID, 'relation': rel})

    return {'nodes': nodes, 'edges': edges}
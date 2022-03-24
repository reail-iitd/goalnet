from .constants import *
from sentence_transformers import SentenceTransformer
import nltk
import re
import numpy as np
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cpu')
stop_words = set({"the","a","and","then","or","next","first"})

def remove_braces(s):
    return s.replace('(', '').replace(')', '')

def dense_vector(obj):
    if obj in VOCAB_VECT:
        return VOCAB_VECT[obj]
    if obj in conceptnet_vectors:
        return conceptnet_vectors[obj]
    raise Exception("No dense representation found for: " + obj)

def preprocess(sent):
    #remove punctuation
    sent = re.sub('[^A-Za-z]+', ' ', sent)
    #lowercase and remove stop words
    processed_sent = ' '.join(e.lower() for e in sent.split() if e.lower() not in stop_words)
    return processed_sent

def goalObjEmbedArgMap(sent, argnouns):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = nltk.word_tokenize(preprocess(sent))
    nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    embed = []
    ######## without argument mapping ########
    #for word in argnouns:
    #    if word in VOCAB_VECT or word in conceptnet_vectors:
    #        embed.append(np.asarray(dense_vector(word)))
    ######## without argument mapping ########
    for word in nouns:
        embed.append(np.asarray(dense_vector(word)))
    return embed

def goalObjEmbed(sent):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = nltk.word_tokenize(preprocess(sent))
    nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    if len(nouns) == 0: nouns = tokenized
    embed = []
    for word in nouns:
        embed.append(np.asarray(dense_vector(word)))
    return embed

def form_goal_vec_sBERT(text):
    return sBERT_model.encode(text)

def dense_vector_embed(sent):
    sent = preprocess(sent)
    embed = np.asarray([0.0] * PRETRAINED_VECTOR_SIZE)
    obj_count = 0
    for obj in sent:
        if obj in VOCAB_VECT:
            embed += VOCAB_VECT[obj]
            obj_count += 1
        elif obj in conceptnet_vectors:
            embed += conceptnet_vectors[obj]
            obj_count += 1
        # else:
        #     print(obj, " - no dense embedding found for it!")
    return embed/max(1, obj_count)

# get environment domain based on the object set (for masking)
def get_env_domain(state):
    _, objects = get_environment(state)
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    return 'kitchen' if inter1 > inter2 else 'living'

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
    return environ, objects

# environment = (dict of relation:[(obj11, obj12), (obj21, obj22)......], objects list)
def get_graph(state):
    env, objects = get_environment(state)
    nodes = []

    for obj in all_objects:
        node = {}
        node['populate'] = obj in objects
        node["id"] = all_objects.index(obj)
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
            fromID = all_objects.index(remove_braces(t[0]))
            toID = all_objects.index(remove_braces(t[1]))
            edges.append({'from': fromID, 'to': toID, 'relation': rel})
    for i, obj in enumerate(all_objects):
        edges.append({'from': i, 'to': i, 'relation': 'Empty'})

    return {'nodes': nodes, 'edges': edges}
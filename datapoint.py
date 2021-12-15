import pickle
import dgl
import numpy as np
import os
import torch
import random
from tqdm import tqdm
import nltk
# nltk.download('punkt')

import re

from constants import *
from sentence_transformers import SentenceTransformer
sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cpu')
# ~/.conda/envs/py35/lib/python3.6/site-packages/sentence_transformers

# from model import *
# word_embed_size = 384
word_embed_size = PRETRAINED_VECTOR_SIZE

def get_env_objects(objects):
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    if inter1 > inter2:
        objects = all_objects_kitchen
    else:
        objects = all_objects_living
    return objects


def get_environment(states):
    environ = {}
    objects = set()
    for rel in all_relations:
        environ[rel] = []

    for s in states:
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

    # if "Robot" in objects: objects.remove("Robot")
    # objects.append("Robot")
    # return environ, list(set(get_env_objects(objects)).union(set(objects)))
    return environ, get_env_objects(objects)

def word_clean(word):
    word = word.split("_")[0]
    l = re.sub( r"([A-Z])", r" \1", word).split()
    return ' '.join(l)

def remove_braces(s):
    if s[0] == '(':
        s = s[1:]
    if s[len(s) - 1] == ')':
        s = s[0:-1]
    return s


def dense_vector(vector, object):
    if object in vector.keys():
        return vector[object]
    if object in conceptnet_vectors.keys():
        return conceptnet_vectors[object]
    else:
        return np.asarray([0] * PRETRAINED_VECTOR_SIZE)
    raise Exception("No dense representation found for: " + object)

def form_goal_vec_sBERT(text):
    return(sBERT_model.encode(text))

# def dense_vector(vector, object):
#     obj = word_clean(object)
#     return form_goal_vec_sBERT(obj)

# def embed_state(vector, state):
#     words = re.findall('[A-Z][^A-Z]*', state)
#     embed = np.zeros(word_embed_size)
#     for w in words:
#         if not (w.lower() == "is"):
#             embed += np.asarray(dense_vector(vector, w.lower()))
#     return embed  # not averaging !!


class Datapoint:
    def __init__(self, sent="", initial_state=set(), final_state=set(), delta_g=set(), delta_g_inv=set(), action_seq=[], file_name=""):
        self.sent = sent
        # self.objects = objects
        self.initial_state = initial_state
        self.final_state = final_state
        self.delta_g = delta_g
        self.delta_g_inv = delta_g_inv
        self.action_seq = action_seq
        self.file_name = file_name

    def save_point(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output)

    def load_point(self, filename):
        with open(filename, "rb") as input:
            input.seek(0)
            tmp = pickle.load(input)
            self.sent = tmp.sent
            # self.objects = tmp.objects
            self.initial_state = tmp.initial_state
            self.final_state = tmp.final_state
            self.delta_g = tmp.delta_g
            self.delta_g_inv = tmp.delta_g_inv
            self.action_seq = tmp.action_seq if hasattr(tmp, 'action_seq') else []
            self.file_name = tmp.file_name if hasattr(tmp, 'file_name') else ""

    def print_point(self):
        print(self.sent + "\n" + "\n Objects \n" + "\n Initial state \n" + str(
            self.initial_state) + "\n Final State \n" + str(self.final_state) + "\n delta_g \n" + str(
            self.delta_g) + "\n delta_g_inv \n" + str(self.delta_g_inv) + "\n")

    def goalObj_embed(self):
        is_noun = lambda pos: pos[:2] == 'NN'
        # or pos[:2] == 'VB'
        tokenized = nltk.word_tokenize(self.sent)
        nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        if len(nouns) == 0: nouns = tokenized
        embed = []
        # embed = np.zeros(PRETRAINED_VECTOR_SIZE)
        for word in nouns:
            embed.append(np.asarray(dense_vector(VOCAB_VECT, word)))
            # embed+= np.asarray(dense_vector(VOCAB_VECT, word))
        # return embed/len(nouns)
        return embed

    # environment = (dict of relation:[(obj11, obj12), (obj21, obj22)......], objects list)
    def get_graph(self, environment):
        env, env_objects = environment
        nodes = []

        for obj in env_objects:
            node = {}
            node["id"] = env_objects.index(obj)
            node["name"] = obj
            node["prop"] = all_obj_prop[obj]
            node["vector"] = dense_vector(VOCAB_VECT, obj)
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


# gives set of all datapoint graphs - complete set of datapoints
class DGLDataset():
    def __init__(self, program_dir, embed):
        graphs, lang_embed, delta_g, sents, goalObj_embed, objects, obj_states, nodes_name, initial_env, env_objects, action_seq = \
            [], [], [], [], [], [], [], [], [], [], []
        delta_g_inv = []
        file_name = []
        all_files = list(os.walk(program_dir))
        tmp_cnt = 0
        for path, dirs, files in tqdm(all_files):
            if ("train" in program_dir):
                random.shuffle(files)
            for f in files:
                tmp_cnt += 1
                dp = encode_datapoint(path + "/" + f, embed)
                if dp is None:
                    continue
                graphs.append(dp[0])
                lang_embed.append(dp[1])
                delta_g.append(dp[2])
                sents.append(dp[3])
                goalObj_embed.append(dp[4])
                objects.append(dp[5])
                obj_states.append(dp[6])
                nodes_name.append(dp[7])  # This is sent to model.py to analyse attention weights - these are node names
                initial_env.append(dp[8])
                env_objects.append(dp[9])
                action_seq.append(dp[10])
                delta_g_inv.append(dp[11])
                file_name.append(dp[12])
        self.graphs = graphs
        self.lang = lang_embed
        self.delta_g = delta_g
        self.sents = sents
        self.goalObjectsVec = goalObj_embed
        self.objects = objects
        self.obj_states = obj_states
        self.init = initial_env
        self.nodes_name = nodes_name
        self.env_objects = env_objects
        self.action_seq = action_seq
        self.delta_g_inv = delta_g_inv
        self.file_name = file_name
        # print("Graph len: ", len(graphs))
        if len(graphs) > 0:
            self.features = graphs[0].ndata['feat'].shape[1]
        else:
            self.features = 0
        # self.lang = lang_embed

# coverts a single environment state to dgl graph
def convertToDGLGraph(graph_data):
    """ Converts the graph from the datapoint graph of a state (initial/final/delta_g etc) into a DGL form of graph."""
    # Make edge sets
    near, on, inside, grasp, obj_index = [], [], [], [], []
    for edge in graph_data["edges"]:
        if edge["relation"] == "Near":
            near.append((edge["from"], edge["to"]))
        elif edge["relation"] == "In":
            inside.append((edge["from"], edge["to"]))
        elif edge["relation"] == "On":
            on.append((edge["from"], edge["to"]))
        # elif edge["relation"] == "state": state.append((edge["from"], edge["to"]))
        elif edge["relation"] == "Grasping":
            grasp.append((edge["from"], edge["to"]))

    n_nodes = len(graph_data["nodes"])

    edgeDict = {
        ('object', 'Near', 'object'): near,
        ('object', 'In', 'object'): inside,
        ('object', 'On', 'object'): on,
        # ('object', 'state', 'object'): state,
        ('object', 'Grasping', 'object'): grasp,
        ('object', 'Agent', 'object'): [(n_nodes - 1, n_nodes - 1)],
    }

    g = dgl.heterograph(edgeDict)
    # Add node features
    # node_states = torch.zeros([n_nodes, len(all_fluents)], dtype=torch.float) # State vector
    node_states = torch.zeros([n_nodes, word_embed_size], dtype=torch.float)
    node_prop = torch.zeros([n_nodes, len(all_non_fluents)], dtype=torch.float)  # State vector
    node_vectors = torch.zeros([n_nodes, word_embed_size], dtype=torch.float)  # Conceptnet embedding
    for i, node in enumerate(graph_data["nodes"]):
        states = node["state_var"]
        prop = node["prop"]
        node_id = node["id"]
        count = 0
        for state in states:
            # idx = all_fluents.index(state)
            # node_states[node_id, idx] = 1   # node_states is a 2D matrix of objects * states
            # print(state, ": \n", dense_vector(VOCAB_VECT, state), "\n")
            node_states[node_id] += torch.tensor((dense_vector(VOCAB_VECT, state).tolist()))
            count += 1
        node_states[node_id] = node_states[node_id] / max(count, 1)
        for state in prop:
            if len(state) > 0:
                idx = all_non_fluents.index(state)
            else:
                continue
            node_prop[node_id, idx] = 1  # node_states is a 2D matrix of objects * states

        node_vectors[node_id] = torch.FloatTensor(node["vector"])

    g.ndata['feat'] = torch.cat((node_vectors, node_states, node_prop), 1)
    return g

# gives a set of dgl graphs for a single datapoint - initial_env, delta_g, delta_g_inv
def getDGLGraph(pathToDatapoint):
    dp = Datapoint()
    dp.load_point(pathToDatapoint)
    # Initial Graph
    graph_init = convertToDGLGraph(dp.get_graph(get_environment(dp.initial_state)))
    graph_delta = convertToDGLGraph(dp.get_graph(get_environment(dp.delta_g)))
    graph_delta_inv = convertToDGLGraph(dp.get_graph(get_environment(dp.delta_g_inv)))
    return (graph_init, graph_delta, graph_delta_inv)


def form_goal_vec(data, text):
    goal_vec = np.zeros(word_embed_size)
    count = 0
    for j in text.split():
        #  goal_vec += dense_vector(data, j)
        # if not(j=="platesa" or j=="stov" or j=="nourishments"):
        if j in data.keys() and len(data[j]) > 0:
            goal_vec += np.asarray(data[j])
            count += 1

    goal_vec /= max(count, 1)
    return goal_vec


# returns a complete encoding of initial_state to dgl graph and language embedding for a datapoint
def encode_datapoint(pathToDatapoint, embed):
    dp = Datapoint()
    dp.load_point(pathToDatapoint)
    # sentence_embed = form_goal_vec_sBERT(dp.sent)
    #print(sen_embed_new)
    #print(sen_embed_new.shape)
    #kskcjs
    sentence_embed = form_goal_vec(embed, dp.sent)
    environment = get_environment(dp.initial_state)
    graph_tmp = dp.get_graph(environment)
    graph_init = convertToDGLGraph(graph_tmp)
    nodes_name = []
    # print("Size = ", graph_init.ndata["feat"].size())
    for node in graph_tmp['nodes']:
        nodes_name.append(node['name'])

    env, env_object = environment

    # get all possible states for objects
    env_states = []
    for obj in env_object:
        try:
            env_states.extend(list(all_object_states[obj]))
        except:
            continue
    env_states = list(set(env_states))
    # convert env objects to one hot vector
    env_obj_arr = np.zeros(N_objects)
    for obj in env_object:
        env_obj_arr[all_objects.index(obj)] = 1
    # convert env states to one hot vector
    env_state_arr = np.zeros(N_fluents)
    for state in env_states:
        env_state_arr[all_fluents.index(state)] = 1
    if len(dp.delta_g) > 0:
        return (
        graph_init, sentence_embed, dp.delta_g, dp.sent, dp.goalObj_embed(), env_obj_arr, env_state_arr, nodes_name,
        dp.initial_state, env_object, dp.action_seq, dp.delta_g_inv, dp.file_name)
    else:
        return None


# print(encode_datapoint('Type6_Final_train_test_0902_SOPPOS_2/train_augment_edited/0', conceptnet_vectors)[0].ndata['feat'].shape[1])
# dp = Datapoint()
# dp.load_point('../train_acl/147_144.pkl')
# print(dp.print_point())

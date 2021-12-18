import pickle
import dgl
import numpy as np
import os
import torch
import random
import nltk
import re
from constants import *
from sentence_transformers import SentenceTransformer
# from datapoint import convertToDGLGraph

sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cpu')

def get_env_objects(objects):
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    if inter1 > inter2:
        objects = all_objects_kitchen
    else:
        objects = all_objects_living
    return objects

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
    raise Exception("No dense representation found for: " + object)

def form_goal_vec_sBERT(text):
    return(sBERT_model.encode(text))

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
    return environ, get_env_objects(objects)

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
        self.graph = []
        self.sent_embed = []
        self.goal_obj_embed = []

    def save_point(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output)

    def load_point(self, filename):
        with open(filename, "r") as fh:
            tmp = json.load(fh)
        self.sent = tmp['sent']
        self.initial_state = tmp['initial_state']
        self.final_state = tmp['final_state']
        self.delta_g = tmp['delta_g']
        self.delta_g_inv = tmp['delta_g_inv']
        self.action_seq = tmp['action_seq'] if 'action_seq' in tmp else []
        self.file_name = tmp['file_name'] if 'file_name' in tmp else ""

    def print_point(self):
        print(self.sent + "\n" + "\n Objects \n" + "\n Initial state \n" + str(
            self.initial_state) + "\n Final State \n" + str(self.final_state) + "\n delta_g \n" + str(
            self.delta_g) + "\n delta_g_inv \n" + str(self.delta_g_inv) + "\n")

    def goalObjEmbed(self):
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

    # coverts a single environment state to dgl graph
    def convertToDGLGraph(self, graph_data):
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
            ('object', 'Grasping', 'object'): grasp,
        }
        g = dgl.heterograph(edgeDict)
        # Add node features
        node_prop = torch.zeros([n_nodes, len(all_non_fluents)], dtype=torch.float)  # Non-fluent vector
        node_fluents = torch.zeros([n_nodes, MAX_REL], dtype=torch.float) # fluent states
        node_vectors = torch.zeros([n_nodes, word_embed_size], dtype=torch.float)  # Conceptnet embedding
        for i, node in enumerate(graph_data["nodes"]):
            states = node["state_var"]
            prop = node["prop"]
            node_id = node["id"]
            count = 0
            for state in states:
                idx = all_object_states[node["name"]].index(state)
                node_fluents[node_id][idx] = 1   # node_states is a 2D matrix of objects * states
            # node_states[node_id] = node_states[node_id] / max(count, 1)
            for state in prop:
                if len(state) > 0:
                    idx = all_non_fluents.index(state)
                else:
                    continue
                node_prop[node_id][idx] = 1  # node_states is a 2D matrix of objects * states
            node_vectors[node_id] = torch.FloatTensor(node["vector"])
        g.ndata['feat'] = torch.cat((node_vectors, node_fluents, node_prop), 1)
        return g

    # returns a complete encoding of initial_state to dgl graph and language embedding for a datapoint
    def encode_datapoint(self):
        self.sent_embed  = form_goal_vec_sBERT(self.sent)
        environment = get_environment(self.initial_state)
        graph_tmp = self.get_graph(environment)
        self.graph = self.convertToDGLGraph(graph_tmp)
        self.goal_obj_embed = self.goalObjEmbed()

        _, env_object = environment
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

        self.objects = env_obj_arr
        self.obj_states = env_state_arr
        self.env_objects = env_object
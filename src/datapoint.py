import pickle
import dgl
import numpy as np
import os
import torch
import random
import re
from copy import deepcopy
from utils.helper import *
from utils.util import *

class Datapoint:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_point(file_path)
        self.sent_embed = []
        self.goal_obj_embed = []
        self.encode_datapoint()

    def load_point(self, file_path):
        with open(file_path, "r") as fh:
            tmp = json.load(fh)
        self.sent = tmp['sent']
        self.file_name = tmp['filename'] if 'filename' in tmp else ""
        self.states = tmp['initial_states']
        self.state_dict = deepcopy(tmp['initial_states'])
        for i, s_dict in enumerate(self.state_dict):
            self.state_dict[i] = [rel for rel in s_dict if len(rel.split()) == 3]
        self.arg_map = tmp['arg_mapping'] if 'arg_mapping' in tmp else None
        self.goal_objects = []
        if self.arg_map: # ground objects in the sent instruction
            for arg, mapping in self.arg_map.items(): 
                self.arg_map[arg] = [noun.replace('(', '').replace(')', '') for noun in mapping]
                self.goal_objects += [noun.replace('(', '').replace(')', '') for noun in mapping]
        self.delta_g = tmp['delta_g']
        self.delta_g_inv = tmp['delta_g_inv']
        self.action_seq = tmp['action_seq'] if 'action_seq' in tmp else []

    # coverts a single environment state to dgl graph
    def convertToDGLGraph(self, state):
        """ Converts the graph from the datapoint graph of a state (initial/final/delta_g etc) into a DGL form of graph."""
        graph_data = get_graph(state)
        # Make edge sets
        near, on, inside, grasp, empty = [], [], [], [], []
        for edge in graph_data["edges"]:
            if edge["relation"] == "Near":
                near.append((edge["from"], edge["to"]))
            elif edge["relation"] == "In":
                inside.append((edge["from"], edge["to"]))
            elif edge["relation"] == "On":
                on.append((edge["from"], edge["to"]))
            elif edge["relation"] == "Grasping":
                grasp.append((edge["from"], edge["to"]))
            elif edge["relation"] == "Empty":
                empty.append((edge["from"], edge["to"]))
        n_nodes = len(graph_data["nodes"])
        edgeDict = {
            ('object', 'Near', 'object'): near,
            ('object', 'In', 'object'): inside,
            ('object', 'On', 'object'): on,
            ('object', 'Grasping', 'object'): grasp,
            ('object', 'Empty', 'object'): empty,
        }
        g = dgl.heterograph(edgeDict)
        # Add node features
        node_prop = torch.zeros([n_nodes, len(all_non_fluents)], dtype=torch.float)  # Non-fluent vector
        node_fluents = torch.zeros([n_nodes, MAX_REL], dtype=torch.float) # fluent states
        node_vectors = torch.zeros([n_nodes, word_embed_size], dtype=torch.float)  # Conceptnet embedding
        adj_matrix = torch.zeros([n_nodes, N_objects], dtype=torch.float)  # Relations with other adjacent nodes
        
        for edge in graph_data["edges"]:
            if not edge['relation'] == "Empty":
                adj_matrix[edge["from"]][edge["to"]] = 1

        for i, node in enumerate(graph_data["nodes"]):
            if not node['populate']: continue
            states = node["state_var"]
            prop = node["prop"]
            node_id = node["id"]
            count = 0
            for state in states:
                idx = all_object_states[node["name"]].index(state)
                node_fluents[node_id][idx] = 1   # node_states is a 2D matrix of objects * states
            for state in prop:
                if len(state) > 0:
                    idx = all_non_fluents.index(state)
                    node_prop[node_id][idx] = 1  # node_states is a 2D matrix of objects * states
            node_vectors[node_id] = torch.FloatTensor(node["vector"])
        # feat_mat = torch.cat((node_vectors, node_fluents, node_prop, node_vectors), 1)
        g.ndata['feat'] = torch.cat((node_vectors, node_fluents, node_prop), 1)
        return g, adj_matrix 

    # returns a complete encoding of initial_state to dgl graph and language embedding for a datapoint
    def encode_datapoint(self):
        self.env_domain = get_env_domain(self.states[0])
        self.sent_embed  = torch.tensor(form_goal_vec_sBERT(self.sent), dtype=torch.float)
        # self.sent_embed  = torch.tensor(dense_vector_embed(self.sent), dtype=torch.float)
        self.adj_matrix = [self.convertToDGLGraph(state)[1] for state in self.states]

        self.states = [self.convertToDGLGraph(state)[0] for state in self.states]
        
        self.goal_obj_embed = torch.tensor(goalObjEmbedArgMap(self.sent, self.goal_objects), dtype=torch.float)
        if 0 in self.goal_obj_embed.shape:
            self.goal_obj_embed = torch.zeros(1, PRETRAINED_VECTOR_SIZE)
        self.delta_g_embed = [string2vec(i[:1]) for i in self.delta_g]
        self.delta_g_inv_embed = [string2vec(i[:1]) for i in self.delta_g_inv]
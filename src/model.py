import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase
import dgl.function as fn
from .GNN import *
from .HAN import *
from utils.constants import *
from .datapoint import *
from dgl.nn import GraphConv
from sentence_transformers import SentenceTransformer

class Simple_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(Simple_Model, self).__init__()
        self.name = "Simple_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_embed = nn.Sequential(nn.Linear(in_feats, n_hidden), self.activation)
        self.goal_obj_attention = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Softmax(dim=0))
        self.fc = nn.Sequential(nn.Linear(n_hidden * 4, n_hidden), self.activation)
        self.lstm = nn.LSTM(n_hidden, n_hidden)
        self.action = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1 = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2 = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

    def forward(self, g, goalVec, goalObjectsVec, lstm_hidden=None):
        # embed graph, goal vec based attention
        h = g.ndata['feat']
        goal_embed = self.embed_sbert(goalVec)
        attn_weights = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights.t(), h)
        h_embed = self.graph_embed(h_embed.view(-1))

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # concatenate goal purpose embedding
        lstm_h = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden)) if lstm_hidden is None else lstm_hidden
        h_hist, lstm_hidden = self.lstm(h_embed.view(1,1,-1), lstm_hidden)
        final_to_decode = self.fc(torch.cat([h_embed, h_hist.view(-1), goal_obj_embed, goal_embed]))

        # head 1 (delta_g)
        action = self.action(final_to_decode)

        one_hot_action = [0 for _ in range(action.shape[0])]
        one_hot_action[torch.argmax(action).item()] = 1
        one_hot_action = torch.Tensor(one_hot_action)

        pred1_object = self.obj1(torch.cat([final_to_decode, one_hot_action]))

        one_hot_pred1 = [0 for _ in range(pred1_object.shape[0])]
        one_hot_pred1[torch.argmax(pred1_object).item()] = 1 
        one_hot_pred1 = torch.Tensor(one_hot_pred1)

        pred2_object = self.obj2(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        pred2_state = self.state(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        return action, pred1_object, pred2_object, pred2_state, lstm_hidden

class GGCN_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(GGCN_Model, self).__init__()
        self.name = "GGCN_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=self.activation))
        for i in range(2):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=self.activation))
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(n_hidden + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_embed = nn.Sequential(nn.Linear(n_hidden, n_hidden), self.activation)
        self.goal_obj_attention = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Softmax(dim=0))
        self.fc = nn.Sequential(nn.Linear(n_hidden * 4, n_hidden), self.activation)
        self.lstm = nn.LSTM(n_hidden, n_hidden)
        self.action = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1 = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2 = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

    def forward(self, g, goalVec, goalObjectsVec, lstm_hidden=None):
        # embed graph, goal vec based attention
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g,h)     #applied ggcn
        goal_embed = self.embed_sbert(goalVec)
        attn_weights = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights.t(), h)
        h_embed = self.graph_embed(h_embed.view(-1))

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # concatenate goal purpose embedding
        lstm_h = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden)) if lstm_hidden is None else lstm_hidden
        h_hist, lstm_hidden = self.lstm(h_embed.view(1,1,-1), lstm_hidden)
        final_to_decode = self.fc(torch.cat([h_embed, h_hist.view(-1), goal_obj_embed, goal_embed]))

        # head 1 (delta_g)
        action = self.action(final_to_decode)

        one_hot_action = [0 for _ in range(action.shape[0])]
        one_hot_action[torch.argmax(action).item()] = 1
        one_hot_action = torch.Tensor(one_hot_action)

        pred1_object = self.obj1(torch.cat([final_to_decode, one_hot_action]))

        one_hot_pred1 = [0 for _ in range(pred1_object.shape[0])]
        one_hot_pred1[torch.argmax(pred1_object).item()] = 1 
        one_hot_pred1 = torch.Tensor(one_hot_pred1)

        pred2_object = self.obj2(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        pred2_state = self.state(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        return action, pred1_object, pred2_object, pred2_state, lstm_hidden

class HAN_model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(HAN_model, self).__init__()
        self.name = "HAN_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths=[['Near', 'In', 'On', 'Grasping', "Empty"]], in_size=in_feats, out_size=n_hidden, layer_num_heads=3, dropout=0.5))
        self.layers.append(HANLayer([['Near', 'In', 'On', 'Grasping', "Empty"]], n_hidden * 3, n_hidden, 3, 0.5))
        self.layers.append(HANLayer([['Near', 'In', 'On', 'Grasping', "Empty"]], n_hidden * 3, n_hidden, 1, 0.5))

        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(n_hidden + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_embed = nn.Sequential(nn.Linear(n_hidden, n_hidden), self.activation)
        self.goal_obj_attention = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Softmax(dim=0))
        self.fc = nn.Sequential(nn.Linear(n_hidden * 4, n_hidden), self.activation)
        self.lstm = nn.LSTM(n_hidden, n_hidden)
        self.action = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1 = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2 = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

    def forward(self, g, goalVec, goalObjectsVec, lstm_hidden=None):
        # embed graph, goal vec based attention
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g,h)     #applied ggcn
        goal_embed = self.embed_sbert(goalVec)
        attn_weights = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights.t(), h)
        h_embed = self.graph_embed(h_embed.view(-1))

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # concatenate goal purpose embedding
        lstm_h = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden)) if lstm_hidden is None else lstm_hidden
        h_hist, lstm_hidden = self.lstm(h_embed.view(1,1,-1), lstm_hidden)
        final_to_decode = self.fc(torch.cat([h_embed, h_hist.view(-1), goal_obj_embed, goal_embed]))

        # head 1 (delta_g)
        action = self.action(final_to_decode)

        one_hot_action = [0 for _ in range(action.shape[0])]
        one_hot_action[torch.argmax(action).item()] = 1
        one_hot_action = torch.Tensor(one_hot_action)

        pred1_object = self.obj1(torch.cat([final_to_decode, one_hot_action]))

        one_hot_pred1 = [0 for _ in range(pred1_object.shape[0])]
        one_hot_pred1[torch.argmax(pred1_object).item()] = 1 
        one_hot_pred1 = torch.Tensor(one_hot_pred1)

        pred2_object = self.obj2(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        pred2_state = self.state(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        return action, pred1_object, pred2_object, pred2_state, lstm_hidden
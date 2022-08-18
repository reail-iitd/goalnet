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

class GoalNet_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(GoalNet_Model, self).__init__()
        self.name = "GoalNet_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        # self.embed_sbert = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_adj = nn.Sequential(nn.Linear(n_objects, 1), nn.Sigmoid())
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_embed = nn.Sequential(nn.Linear(in_feats, n_hidden), self.activation)
        self.goal_obj_attention = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Softmax(dim=0))
        self.fc = nn.Sequential(nn.Linear(n_hidden * 4, n_hidden), self.activation)
        self.pred_lstm = nn.LSTM((len(all_relations) + 1 + 2 * PRETRAINED_VECTOR_SIZE), n_hidden)
        self.action = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1 = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2 = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))
        self.action_inv = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1_inv = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2_inv = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state_inv = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

    def forward(self, g, goalVec, goalObjectsVec, pred, lstm_hidden=None):
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

        # feed predicted constraint in LSTM
        lstm_h = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden)) if lstm_hidden is None else lstm_hidden
        pred_hist, lstm_hidden = self.pred_lstm(torch.cat([string2embed(pred)]).view(1,1,-1), lstm_h)
        final_to_decode = self.fc(torch.cat([h_embed, goal_obj_embed, goal_embed, pred_hist.view(-1)]))

        # head 1 (delta_g)
        action = self.action(final_to_decode)
        one_hot_action = gumbel_softmax(action, 0.01)
        pred1_object = self.obj1(torch.cat([final_to_decode, one_hot_action]))
        one_hot_pred1 = gumbel_softmax(pred1_object, 0.01)
        pred2_object = self.obj2(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))
        pred2_state = self.state(torch.cat([final_to_decode, one_hot_action, one_hot_pred1]))

        # head 2 (delta_g_inv)
        action_inv = self.action_inv(final_to_decode)
        one_hot_action_inv = gumbel_softmax(action_inv, 0.01)
        pred1_object_inv = self.obj1_inv(torch.cat([final_to_decode, one_hot_action_inv]))
        one_hot_pred1_inv = gumbel_softmax(pred1_object_inv, 0.01)
        pred2_object_inv = self.obj2_inv(torch.cat([final_to_decode, one_hot_action_inv, one_hot_pred1_inv]))
        pred2_state_inv = self.state_inv(torch.cat([final_to_decode, one_hot_action_inv, one_hot_pred1_inv]))

        return (action, pred1_object, pred2_object, pred2_state, action_inv, pred1_object_inv, pred2_object_inv, pred2_state_inv), lstm_hidden
        # return (action, pred1_object, pred2_object, pred2_state), lstm_hidden

class GoalNet_Star_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(GoalNet_Star_Model, self).__init__()
        self.name = "GoalNet_Star_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        #self.gcn = HeteroRGCNLayer(in_feats, in_feats, etypes, activation=self.activation)
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        # self.embed_sbert = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.embed_pos = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_embed = nn.Sequential(nn.Linear(in_feats, n_hidden), self.activation)
        self.goal_obj_attention = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Softmax(dim=0))
        self.fc = nn.Sequential(nn.Linear(n_hidden * 4, n_hidden), self.activation)
        self.pred_lstm = nn.LSTM((len(all_relations) + 1 + 2 * PRETRAINED_VECTOR_SIZE), n_hidden)
        
        self.action = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1 = nn.Sequential(nn.Linear(n_hidden*2 + len(all_relations) + 1 + 1, n_hidden),self.activation,nn.Linear(n_hidden,1))
        self.obj2 = nn.Sequential(nn.Linear(n_hidden*2 + len(all_relations) + 1 + 1 + 1, n_hidden), self.activation, nn.Linear(n_hidden,1))
        #self.state = nn.Sequential(nn.Linear(n_hidden*3 + len(all_relations) + 1 + 1, n_hidden),self.activation, nn.Linear(n_hidden,1))
        self.state = nn.Sequential(nn.Linear(n_hidden + n_hidden + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

        self.action_inv = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1_inv = nn.Sequential(nn.Linear(n_hidden*2 + len(all_relations) + 1 + 1 , n_hidden),self.activation,nn.Linear(n_hidden,1))
        self.obj2_inv = nn.Sequential(nn.Linear(n_hidden*2 + len(all_relations) + 1 + 1 +1 , n_hidden), self.activation, nn.Linear(n_hidden,1))
        #self.state_inv = nn.Sequential(nn.Linear(n_hidden*3 + len(all_relations) + 1 + 1, n_hidden),self.activation, nn.Linear(n_hidden,1))
        self.state_inv = nn.Sequential(nn.Linear(n_hidden + n_hidden + len(all_relations) + 1, n_states), nn.Softmax(dim=0))
        
        self.n_objects = n_objects
        self.in_feats = in_feats
        #self.n_states = n_states
        l = []
        pos = []
        for obj in universal_objects:
            l.append(dense_vector(obj))
            instance = obj.split("_")[-1]
            inst_arr = [0,0,0,0]
            if(instance.isnumeric()):
                inst_arr[int(instance)-1] = 1
            pos.append(inst_arr)
        self.object_vec = torch.Tensor(l)
        self.object_pos = torch.Tensor(pos)


    def forward(self, g, goalVec, goalObjectsVec, pred, lstm_hidden=None):
        # embed graph, goal vec based attention
        h = g.ndata['feat']  #GNN   
        
        #h = torch.cat((h_vec, h_adj), 1)  
        goal_embed = self.embed_sbert(goalVec) 
        attn_weights = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights.t(), h)
        h_embed = self.graph_embed(h_embed.view(-1))

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # feed predicted constraint in LSTM
        lstm_h = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden)) if lstm_hidden is None else lstm_hidden
        pred_hist, lstm_hidden = self.pred_lstm(torch.cat([string2embed(pred)]).view(1,1,-1), lstm_h)
        final_to_decode = self.fc(torch.cat([h_embed, goal_obj_embed, goal_embed, pred_hist.view(-1)]))

        # head 1 (delta_g)
        action = self.action(final_to_decode)
        one_hot_action = gumbel_softmax(action, 0.01)
        pred1_input = torch.cat([final_to_decode, one_hot_action],0)
        pred1_object = self.obj1(
                        torch.cat([pred1_input.repeat(self.n_objects).view(self.n_objects,-1), 
                                   self.activation(self.embed(self.object_vec)), 
                                   self.embed_pos(self.object_pos)], 1)) 
                        #n_obj * 1 .
        pred1_object = torch.sigmoid(pred1_object)
        
        pred2_input = torch.cat([final_to_decode, one_hot_action], 0)
        pred2_object = self.activation(self.obj2(
                        torch.cat([pred2_input.repeat(self.n_objects).view(self.n_objects, -1), #T
                                   self.activation(self.embed(self.object_vec)), 
                                   self.embed_pos(self.object_pos),
                                   pred1_object.view(self.n_objects, 1)], 1)))
        pred2_object = torch.sigmoid(pred2_object)

        #one_hot_pred1 = gumbel_softmax(pred1_object, 0.01)
        pred1_obj_embed = self.activation(self.embed(torch.tensor(dense_vector(universal_objects[torch.argmax(pred1_object)]))))
        pred2_state = self.state(torch.cat([final_to_decode, one_hot_action, pred1_obj_embed]))
    
        # head 2 (delta_g_inv)
        action_inv = self.action_inv(final_to_decode)
        one_hot_action_inv = gumbel_softmax(action_inv, 0.01)

        pred1_input_inv = torch.cat([final_to_decode, one_hot_action_inv],0)
        pred1_object_inv = self.obj1_inv(
                        torch.cat([pred1_input_inv.view(-1).repeat(self.n_objects).view(self.n_objects, -1), 
                                   self.activation(self.embed(self.object_vec)),self.embed_pos(self.object_pos),], 1))
        pred1_object_inv = torch.sigmoid(pred1_object_inv)
        
        
        pred2_input_inv = torch.cat([final_to_decode, one_hot_action_inv], 0)
        pred2_object_inv = self.activation(self.obj2_inv(
                        torch.cat([pred2_input_inv.view(-1).repeat(self.n_objects).view(self.n_objects, -1), 
                                   self.activation(self.embed(self.object_vec)), self.embed_pos(self.object_pos),
                                   pred1_object_inv.view(self.n_objects, 1)], 1)))
        pred2_object_inv = torch.sigmoid(pred2_object_inv)

        #one_hot_pred1_inv = gumbel_softmax(pred1_object_inv, 0.01)
        pred1_obj_inv_embed = self.activation(self.embed(torch.tensor(dense_vector(universal_objects[torch.argmax(pred1_object_inv)]))))
        pred2_state_inv = self.state_inv(torch.cat([final_to_decode, one_hot_action_inv, pred1_obj_inv_embed]))

        return (action, torch.squeeze(pred1_object), torch.squeeze(pred2_object), torch.squeeze(pred2_state), action_inv, torch.squeeze(pred1_object_inv), torch.squeeze(pred2_object_inv), torch.squeeze(pred2_state_inv)), lstm_hidden
        # return (action, pred1_object, pred2_object, pred2_state), lstm_hidden


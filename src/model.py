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
        # self.embed_sbert = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
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
        self.action_inv = nn.Sequential(nn.Linear(n_hidden, len(all_relations) + 1), nn.Softmax(dim=0)) # +1 for null delta_g
        self.obj1_inv = nn.Sequential(nn.Linear(n_hidden + len(all_relations) + 1, n_objects),  nn.Softmax(dim=0))
        self.obj2_inv = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_objects), nn.Softmax(dim=0))
        self.state_inv = nn.Sequential(nn.Linear(n_hidden + n_objects + len(all_relations) + 1, n_states), nn.Softmax(dim=0))

    def forward(self, g, goalVec, goalObjectsVec, pred, lstm_hidden=None):
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
        h_hist, lstm_hidden = self.lstm(h_embed.view(1,1,-1), lstm_h)
        final_to_decode = self.fc(torch.cat([h_embed, h_hist.view(-1), goal_obj_embed, goal_embed]))

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

class GCN_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(GCN_Model, self).__init__()
        self.name = "GCN_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        self.skip_connection = nn.Sequential(nn.Linear(in_feats * 2, 2), nn.Softmax(dim=0))
        self.gcn = HeteroRGCNLayer(in_feats, in_feats, etypes, activation=self.activation)
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_attn_gcn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
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
        h = g.ndata['feat']
        h_gcn = self.gcn(g, h)
        goal_embed = self.embed_sbert(goalVec)

        # embed h using goal attention
        attn_weights_h = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights_h.t(), h).view(-1)

        # embed gcn output using goal attention
        attn_weights_h_gcn = self.graph_attn_gcn(torch.cat([h_gcn, goal_embed.repeat(h.shape[0], 1)], 1))
        h_gcn_embed = torch.mm(attn_weights_h_gcn.t(), h_gcn).view(-1)

        # weighted skip connection
        weights = self.skip_connection(torch.cat([h_embed, h_gcn_embed]))
        h_embed = weights[0] * h_embed + weights[1] * h_gcn_embed
        h_embed = self.graph_embed(h_embed)

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # concatenate goal purpose embedding
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

class HAN_Model(nn.Module):
    def __init__(self,
                 in_feats, 
                 n_hidden,
                 n_objects,
                 n_states,
                 etypes):
        super(HAN_Model, self).__init__()
        self.name = "HAN_Model"
        self.n_hidden = n_hidden
        self.activation = nn.PReLU()
        self.skip_connection = nn.Sequential(nn.Linear(in_feats * 2, 2), nn.Softmax(dim=0))
        self.gcn = HANLayer([['Near'], ['In'], ['On'], ['Grasping'], ["Empty"]], in_feats, in_feats, 1, 0)
        self.embed_sbert = nn.Sequential(nn.Linear(SBERT_VECTOR_SIZE, n_hidden), self.activation)
        self.embed_conceptnet = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation)
        self.graph_attn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
        self.graph_attn_gcn = nn.Sequential(nn.Linear(in_feats + n_hidden, 1), nn.Softmax(dim=1))
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
        h = g.ndata['feat']
        h_gcn = self.gcn(g, h)
        goal_embed = self.embed_sbert(goalVec)

        # embed h using goal attention
        attn_weights_h = self.graph_attn(torch.cat([h, goal_embed.repeat(h.shape[0], 1)], 1))
        h_embed = torch.mm(attn_weights_h.t(), h).view(-1)

        # embed gcn output using goal attention
        attn_weights_h_gcn = self.graph_attn_gcn(torch.cat([h_gcn, goal_embed.repeat(h.shape[0], 1)], 1))
        h_gcn_embed = torch.mm(attn_weights_h_gcn.t(), h_gcn).view(-1)

        # weighted skip connection
        weights = self.skip_connection(torch.cat([h_embed, h_gcn_embed]))
        h_embed = weights[0] * h_embed + weights[1] * h_gcn_embed
        h_embed = self.graph_embed(h_embed)

        # goal conditioned self attention 
        goal_obj_embed = self.embed_conceptnet(goalObjectsVec)
        n_goal_obj = goal_obj_embed.shape[0]
        attn_weights = self.goal_obj_attention(torch.cat([h_embed.repeat(n_goal_obj, 1), goal_obj_embed], 1))
        goal_obj_embed = torch.mm(attn_weights.reshape(1, -1), goal_obj_embed).view(-1)

        # concatenate goal purpose embedding
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

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
from GNN import *
from constants import *
from datapoint import *
from dgl.nn import GraphConv
from sentence_transformers import SentenceTransformer

class GGCN_node_attn_sum(nn.Module):
    def __init__(self,
                 in_feats,  # g.ndata['feat'] = torch.cat((node_vectors, node_states), 1)
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_node_attn_sum, self).__init__()
        self.name = "GGCN_Attn_pairwise_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.activation = nn.PReLU()
        # self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.attention = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        # self.embed2 = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.embed1 = nn.Linear(PRETRAINED_VECTOR_SIZE + 84, n_hidden)
        # self.embed2 = nn.Linear(PRETRAINED_VECTOR_SIZE, 300)
        self.embed3 = nn.Linear(PRETRAINED_VECTOR_SIZE + 84, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + len(all_relations), n_hidden)
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_objects)
        self.p1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        # self.p2 = nn.Linear(n_hidden, n_hidden)
        self.p3 = nn.Linear(n_hidden, len(all_relations))
        self.q1 = nn.Linear(n_hidden + n_hidden + n_objects + len(all_relations), n_hidden)
        # self.q2 = nn.Linear(n_hidden, n_hidden)
        self.q3 = nn.Linear(n_hidden, n_objects + 1)

        self.q1_state = nn.Linear(n_hidden + n_hidden + n_objects + len(all_relations), n_hidden)
        # self.q2_state = nn.Linear(n_hidden, n_hidden)
        self.q3_state = nn.Linear(n_hidden, n_states + 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, goalVec, goalObjectsVec, bool_train, y, epoch, obj_arr=[],state_arr=[],objs=[],outfile=None):
        #if (torch.cuda.is_available()):
        #    g = g.to(torch.device('cuda'))
        # only node embedding - 300 * Num_nodes
        h = g.ndata['feat'][:, :384]
        lang_embed = torch.tensor(goalVec.reshape(1, -1), dtype=torch.float)
        goalObj = torch.tensor(goalObjectsVec, dtype=torch.float)
        #if (torch.cuda.is_available()):
        #    lang_embed = lang_embed.cuda()
        #    goalObj = goalObj.cuda()

        goal_embed = self.activation(self.embed1(lang_embed))  # language input of instruction (only needed)

        # final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        # goalObj_embed = self.activation(self.embed2(goalObj))
        # goalObj_embed = self.embed2(goalObj)

        attn_weights = torch.matmul(goalObj, torch.transpose(h, 0, 1))
        attn_weights = F.softmax(torch.sum(attn_weights, dim=0))
        # attn_weights = F.softmax(torch.sum(F.softmax(torch.matmul(goalObj_embed,torch.transpose(h, 0, 1)),dim=1),dim=0))
        # start code to write attention weights
        if (outfile and len(objs) > 0):
            attn_obj_arr = []
            for i in range(len(objs)):
                attn_obj_arr.append([objs[i], attn_weights[i].item()])
            attn_obj_arr = sorted(attn_obj_arr, key=lambda x: x[1], reverse=True)
            for i in range(len(attn_obj_arr)):
                outfile.write(str(attn_obj_arr[i]) + "\n")

        # attn_weights = torch.tensor(attn_weights.reshape(1, -1))
        scene_embedding = self.embed3(torch.mm(attn_weights.reshape(1, -1), h))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)

        action = self.activation(self.p1(final_to_decode))
        action = self.dropout(action)
        # action = self.activation(self.p2(action))
        action = F.softmax(self.p3(action), dim=1)

        pred_action_values = list(action[0])
        ind_max_action = pred_action_values.index(max(pred_action_values))
        one_hot_action = [0] * len(pred_action_values)
        one_hot_action[ind_max_action] = 1
        one_hot_action = torch.Tensor(one_hot_action).view(1, -1)

        # epsilon = int(epoch/50)+1
        if bool_train and random.random() < (50 / (epoch + 5)):
            one_hot_action = y[None, 0:N_relations]
        #if (torch.cuda.is_available()):
        #    one_hot_action = one_hot_action.cuda()

        pred1_object = self.activation(self.fc1(torch.cat([final_to_decode, one_hot_action], 1)))
        pred1_object = self.dropout(pred1_object)
        # pred1_object = self.fc2(pred1_object)
        pred1_object = self.activation(pred1_object)
        pred1_object = self.fc3(pred1_object)

        '''
        if(ind_max_action == 1 or ind_max_action == 4): #predicted state is Near or Grasping
            pred1_object[0][:] = -10000000#pred1_object.fill_(0)
            pred1_object[0][N_objects - 1] = 1  #set obj1 as Robot
        '''
        pred1_object = F.softmax(pred1_object, dim=1)

        #
        # print(pred1_object)
        pred_action_values = list(action[0])
        pred1_values = list(pred1_object[0])
        ind_max_pred1 = pred1_values.index(max(pred1_values))
        one_hot_pred1 = [0 for _ in range(len(pred1_values))]
        one_hot_pred1[ind_max_pred1] = 1  # one hot for object 1
        one_hot_pred1 = torch.Tensor(one_hot_pred1).view(1, -1)
        # Teahcer forcing till 50 epochs and then randomly with 0.5 probability
        if bool_train and random.random() < (50 / (epoch + 5)):
            one_hot_pred1 = y[None, N_relations:N_relations + N_objects]
        #if (torch.cuda.is_available()):
        #    one_hot_pred1 = one_hot_pred1.cuda()
        pred2_object = self.activation(self.q1(torch.cat([final_to_decode, one_hot_pred1, one_hot_action], 1)))
        pred2_object = self.dropout(pred2_object)
        # pred2_object = self.q2(pred2_object)
        pred2_object = self.activation(pred2_object)
        pred2_object = F.softmax(self.q3(pred2_object), dim=1)

        pred2_state = self.activation(self.q1_state(torch.cat([final_to_decode, one_hot_pred1, one_hot_action], 1)))
        pred2_state = self.dropout(pred2_state)
        # pred2_state = self.q2_state(pred2_state)
        pred2_state = self.activation(pred2_state)
        pred2_state = F.softmax(self.q3_state(pred2_state), dim=1)

        # if action is "state" then pred2_object = 0 and pred2_state != 0 else vice-versa
        return (action, pred1_object, pred2_object, pred2_state)


class GGCN_Attn_pairwise(nn.Module):
    def __init__(self,
                 in_feats, #g.ndata['feat'] = torch.cat((node_vectors, node_states), 1)
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Attn_pairwise, self).__init__()
        self.name = "GGCN_Attn_pairwise_" + str(n_hidden) + "_" + str(n_layers)
        
        # self.conv = GraphConv(in_feats, n_hidden,norm='both', weight=True, bias=True, activation=self.activation, allow_zero_in_degree=True)
        #  etypes=["Near", "In", "On", "Grasping", "Agent"],
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        # self.activation = nn.PReLU()
        # self.attention = nn.Linear(n_hidden + n_hidden, 1)
        # self.attention = nn.Sequential(nn.Linear(n_hidden*2, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        #self.embed2 = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.embed2 = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.embed2.requires_grad = True
        self.embed1 = nn.Linear(PRETRAINED_VECTOR_SIZE+84, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + len(all_relations), n_hidden)
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_objects)
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        # self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, len(all_relations))
        self.q1  = nn.Linear(n_hidden + n_hidden + n_objects + len(all_relations), n_hidden)
        # self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects + 1)
        
        self.q1_state  = nn.Linear(n_hidden + n_hidden + n_objects + len(all_relations), n_hidden)
        # self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states + 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    # def forward(self, g, goalVec, goalObjectsVec, bool_train, y, epoch,objs=[],outfile=None):
    def forward(self, g, goalVec, goalObjectsVec, bool_train, y, epoch,obj_arr=[],state_arr=[],objs=[],outfile=None):
        # if(torch.cuda.is_available()):
        #     g = g.to(torch.device('cuda'))
          #  y = y.cuda()
        # new_g = dgl.metapath_reachable_graph(g, ['Near', 'On', 'In', 'Grasping', "Agent"])
        # h = self.conv(g, g.ndata['feat'])
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g,h)     #applied ggcn

        # goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))  #language input of object
        # # h = torch.sum(h, dim=0).view(1,-1)
        # print("h  ------------------------------------------------------------------------------  ")
        # print(h)
        lang_embed = torch.tensor(goalVec.reshape(1, -1),dtype = torch.float)
        # print("Lang_embed  ------------------------------------------------------------------------------  ")
        # print(lang_embed)
        #goalObj = torch.tensor(goalObjectsVec.reshape(1, -1),dtype = torch.float)
        goalObj = torch.tensor(goalObjectsVec,dtype=torch.float)
        # if(torch.cuda.is_available()):
        #     lang_embed = lang_embed.cuda()
        #     goalObj = goalObj.cuda()

        goal_embed = self.activation(self.embed1(lang_embed))  #language input of instruction (only needed)
        
        # final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        #goalObj_embed = self.activation(self.embed2(goalObj))
        goalObj_embed = self.embed2(goalObj)
       
        #attn_weights = F.softmax(
        #attn_weights = torch.max(F.softmax(
        attn_weights = torch.matmul(goalObj_embed,torch.transpose(h, 0, 1))
        attn_weights = F.softmax(torch.max(attn_weights,dim=0)[0])
        
        #attn_weights = F.softmax(torch.sum(F.softmax(torch.matmul(goalObj_embed,torch.transpose(h, 0, 1)),dim=1),dim=0))
        #start code to write attention weights
        if(outfile and len(objs)>0):
            attn_obj_arr = []
            for i in range(len(objs)):
                attn_obj_arr.append([objs[i],attn_weights[i].item()])
            attn_obj_arr = sorted(attn_obj_arr,key=lambda x: x[1],reverse=True)
            for i in range(10):
                outfile.write(str(attn_obj_arr[i])+"\n")
        
        #end code to write attention weights 
        # attn_weights = torch.tensor(attn_weights.reshape(1,-1))
        scene_embedding = torch.mm(attn_weights.reshape(1,-1), h)
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)

        action = self.activation(self.p1(final_to_decode))
        action = self.dropout(action)
        # action = self.activation(self.p2(action))
        action = F.softmax(self.p3(action), dim=1)

        pred_action_values = list(action[0])
        ind_max_action = pred_action_values.index(max(pred_action_values))
        one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
        one_hot_action = torch.Tensor(one_hot_action).view(1,-1)

        # epsilon = int(epoch/50)+1
        if bool_train and random.random()<(50/(epoch+5)):
            one_hot_action = y[None, 0:N_relations]
        # if(torch.cuda.is_available()):
        #     one_hot_action = one_hot_action.cuda() 
        
        pred1_object = self.activation(self.fc1(torch.cat([final_to_decode, one_hot_action],1)))
        pred1_object = self.dropout(pred1_object)
        # pred1_object = self.fc2(pred1_object)
        pred1_object = self.activation(pred1_object)
        pred1_object = self.fc3(pred1_object)
        pred1_object = F.softmax(pred1_object, dim=1)
        '''
        if(ind_max_action == 1 or ind_max_action == 4): #predicted state is Near or Grasping
            pred1_object[0][:] = -10000000#pred1_object.fill_(0)
            pred1_object[0][N_objects - 1] = 1  #set obj1 as Robot
        '''
           
        #
        # print(pred1_object)
        pred1_values = list(pred1_object[0]); ind_max_pred1 = pred1_values.index(max(pred1_values))
        one_hot_pred1 = [0 for _ in range(len(pred1_values))]; one_hot_pred1[ind_max_pred1] = 1   #one hot for object 1
        one_hot_pred1 = torch.Tensor(one_hot_pred1).view(1,-1)
        # Teahcer forcing till 50 epochs and then randomly with 0.5 probability
        if bool_train and random.random()<(50/(epoch+5)):
            one_hot_pred1 = y[None, N_relations:N_relations+N_objects]
        # if(torch.cuda.is_available()):
        #     one_hot_pred1 = one_hot_pred1.cuda()
        pred2_object = self.activation(self.q1(torch.cat([final_to_decode, one_hot_pred1, one_hot_action],1)))
        pred2_object = self.dropout(pred2_object)
        # pred2_object = self.q2(pred2_object)
        pred2_object = self.activation(pred2_object)
        pred2_object = F.softmax(self.q3(pred2_object), dim=1)

        
        pred2_state = self.activation(self.q1_state(torch.cat([final_to_decode, one_hot_pred1, one_hot_action],1)))
        pred2_state = self.dropout(pred2_state)
        # pred2_state = self.q2_state(pred2_state)
        pred2_state = self.activation(pred2_state)
        pred2_state = F.softmax(self.q3_state(pred2_state), dim=1)

        # if action is "state" then pred2_object = 0 and pred2_state != 0 else vice-versa
        return action, pred1_object, pred2_object, pred2_state
        # return torch.cat((action, pred1_object, pred2_object, pred2_state), 1).flatten()
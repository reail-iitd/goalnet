import math
import numpy as np
from numpy import linalg
import os
from datapoint import Datapoint
from dataset import DGLDataset
from constants import *
import re
from sentence_transformers import SentenceTransformer
sBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
import math


counter = 0
tmp_cnt =0
data_in = "data/"
# dataout = "new_data/"
num_files = 468
verbs = []
test_file_id = [2,13,20,25,40,48,54,56,58,62,70,72,\
                86,88,101,111,113,116,120,121,124,129,\
                132,138,140,143,161,167,171,177,182,186,190,\
                192,201,203,208,212,215,220,235,242,246,\
                249,254,256,257,268,269,273,274,276,278,282,\
                289,299,306,313,314,319,320,323,325,\
                340,346,354,361,362,363,379,380,385,392,405,409,\
                417,422,426,430,434,435,442,443,444,446,\
                450,454,457,460,462,465]
val_count = math.ceil((num_files - len(test_file_id))/5)

train_all_file = open("train_all.txt", "w")
test_all_file = open("test_all.txt", "w")
ref_count = 0
count_delta_g_inv = 0
counter = 0
v_count = 0
for file_index in range(num_files):
    if file_index in test_file_id:
        continue
#     data_all_file = test_all_file if file_index in test_file_id else train_all_file
    print(file_index)
    file_inst = open(data_in+str(file_index)+'.instenv')
    env_states=[]
    action = []
    # read from instenv file
    for line in file_inst:
        tmp = line[0:3]
        if tmp == 'Env':
            env = line[5:-1]
            env_states.append(set(env.split(',')))
        elif tmp == "Ins":
            action.append(line.split(":")[1].strip())
            
    # read from clause file
    start_env_num=[]
    end_env_num=[]
    action_list = []
    # delta_g, delta_g_inv, action = [], [], []
    sent=[]
    file_clause = open(data_in+str(file_index)+'.clauses')
    Lines = file_clause.readlines()
    for i in range(len(Lines)):
        line = Lines[i]
        if line[0:9]=="start env":
            start_env_num.append(int(line.split(':')[1]))
        elif line[0:7] == "end env":
            end_env_num.append(int(line.split(':')[1])+1)
            action_string_tmp = []
            if start_env_num[-1] < len(env_states) and end_env_num[-1] < len(env_states):
                for i in range(start_env_num[-1], end_env_num[-1]):
                    action_string_tmp.append(action[i])
            action_list.append(action_string_tmp)
        elif line[0:4] == "sent":
            instr = line.split(':')[1].strip()
            i+=1
            line = Lines[i]
            while line[0:6] != "clause":
                instr += " " + line.strip()
                i+=1
                line = Lines[i] 
            sent.append(instr)
        elif line[0:11] =="arg mapping": 
            text = sent[-1] if len(sent)>0 else ""
            if "it" in text or "them" in text: #
                print("Text with it or them = ", text)
                args = line.split(':')[1].split(')')[:-1]
                # args = ["(ramen cup,Ramen_1", "(ramen cup,LongCup_1", "(ramen cup,LongCup_2"]
                for a in args:
                    term1 = (a.split(',')[0].replace('(','')).strip()  #term 1 can contains "it"
                    term2 = (a.split(',')[1]).strip().lower().split('_')[0]  #term 2 will contain its mapping
                    if term1 =="it":
                        text = re.sub(r'\bit\b',term2,text)
                    if term1 =="them":
                        text = re.sub(r'\bthem\b',term2,text)
            sent[-1] = text
            
    # write to a folder datapoint
    for i in range(len(sent)):
#         try:
        if end_env_num[i]>start_env_num[i] and len(sent[i])>0:
            delta_g = list(env_states[end_env_num[i]].difference(env_states[start_env_num[i]]))
            delta_g_inv = list(env_states[start_env_num[i]].difference(env_states[end_env_num[i]]))
            if(len(delta_g_inv)==0 and len(delta_g)==0):
                continue
            else:
                data_name = "val_22_11" if v_count<val_count else "train_22_11"
                filename = "data_acl16/"+data_name+"/"+str(file_index)+"_"+str(counter)+".pkl" 
                obj = Datapoint(sent[i],env_states[start_env_num[i]],env_states[end_env_num[i]],delta_g,delta_g_inv, action_list[i], str(file_index)+".pkl")
                obj.save_point(filename)
                v_count += 1
                counter+=1
    break
print("Done writing")
# train_all_file.close()
# test_all_file.close()
print(count_delta_g_inv)
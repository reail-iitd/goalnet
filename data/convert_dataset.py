# converting acl16_original_dataset to json files.

import os
from tqdm import trange
import json, math, re

DIR = './acl16_original_dataset/'
NUM_DATAPOINTS = 469

test_file_id = [2,13,20,25,40,48,54,56,58,62,70,72,\
                86,88,101,111,113,116,120,121,124,129,\
                132,138,140,143,161,167,171,177,182,186,190,\
                192,201,203,208,212,215,220,235,242,246,\
                249,254,256,257,268,269,273,274,276,278,282,\
                289,299,306,313,314,319,320,323,325,\
                340,346,354,361,362,363,379,380,385,392,405,409,\
                417,422,426,430,434,435,442,443,444,446,\
                450,454,457,460,462,465]

val_count = math.ceil((NUM_DATAPOINTS - len(test_file_id))/5)
counter = 0; v_count = 0
for file_index in trange(NUM_DATAPOINTS):
    c_file = f'{DIR}/{file_index}.clauses'
    i_file = f'{DIR}/{file_index}.instenv'
    file_inst = open(i_file, 'r').readlines()
    env_states, action = [], []
    # read from instenv file
    for line in file_inst:
        if 'Env:' in line:
            env_states.append(line.replace('Env: ', '').strip().split(','))
        if 'Instruction:' in line:
            action.append(line.split(":")[1].strip())
    # read from clause file
    start_env_num, end_env_num, action_list, sent = [], [], [], []
    file_clause = open(c_file).readlines()
    for i, line in enumerate(file_clause):
        if "start env:" in line:
            start_env_num.append(int(line.split(':')[1]))
        elif "end env:" in line:
            end_env_num.append(int(line.split(':')[1])+1)
            action_string_tmp = []
            if start_env_num[-1] < len(env_states) and end_env_num[-1] < len(env_states):
                for j in range(start_env_num[-1], end_env_num[-1]):
                    action_string_tmp.append(action[j])
            action_list.append(action_string_tmp)
        elif "sent:" in line:
            instr = line.split(':')[1].strip()
            j = i + 1
            line2 = file_clause[j]
            while "clause" not in line2:
                instr += " " + line2.strip()
                j+=1
                line2 = file_clause[j] 
            sent.append(instr.lower())
        elif "arg mapping:" in line: 
            text = sent[-1] if len(sent)>0 else ""
            if "it" in text or "them" in text: #
                # print("Text with it or them = ", text)
                args = line.split(':')[1].split(')')[:-1]
                for a in args:
                    term1 = (a.split(',')[0].replace('(','')).strip()  #term 1 can contains "it"
                    term2 = (a.split(',')[1]).strip().lower().split('_')[0]  #term 2 will contain its mapping
                    if term1 =="it":
                        text = re.sub(r'\bit\b',term2,text)
                    if term1 =="them":
                        text = re.sub(r'\bthem\b',term2,text)
            sent[-1] = text

    for i in range(len(sent)):
        if end_env_num[i]>start_env_num[i] and len(sent[i])>0:
        	delta_g, delta_g_inv = [], []
        	states = env_states[start_env_num[i]:end_env_num[i]+1]
        	actions = action_list[i] + ['noop']
        	todel = []
        	for j in range(len(states)-1):
        		d_g = list(set(states[j+1]).difference(set(states[j])))
        		d_g_inv = list(set(states[j]).difference(set(states[j+1])))
        		if actions[j] == 'wait': 
        			todel.append(j)
        			continue
        		delta_g.append(d_g); delta_g_inv.append(d_g_inv) 
        	delta_g.append([]); delta_g_inv.append([])
        	for ind in todel[::-1]:
        		del actions[ind]; del states[ind]
        	if(len(delta_g_inv)==0 and len(delta_g)==0): continue
        	data_name = 'test' if file_index in test_file_id else "val" if v_count < val_count else "train"
        	filename = data_name+"/"+str(file_index)+"_"+str(counter)+".json"
        	# print(filename)
        	# print(len(states), len(actions), len(delta_g), len(delta_g_inv))
        	assert len(states) == len(actions) == len(delta_g) == len(delta_g_inv)
        	dp = {
            	'sent': sent[i],
            	'filename': str(file_index),
            	'initial_states': states,
            	'action_seq': actions,
            	'delta_g': delta_g,
            	'delta_g_inv': delta_g_inv
            	}
        	with open(filename, 'w') as fh:
        		json.dump(dp, fh, indent=4)
       		v_count += 1
        	counter+=1

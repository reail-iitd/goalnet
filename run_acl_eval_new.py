# python run_acl_eval_new.py 0808_lossPlan pddl_test_clean_plan test_ied_sji

import subprocess
import pickle
import dgl
import numpy as np
import os
import torch
import random
# from torch.nn import factory_kwargs
from tqdm import tqdm
import nltk
import re
import sys
from datapoint import Datapoint

Score = {}
Score_jac = {}
jac_list, ied_list = [], []
instance_table = {"ramen_all": ['ramen_1', 'instantramen_1'],
                  "stovefireall": ['stovefire1', 'stovefire2', 'stovefire3', 'stovefire4', 'stovefire'],
                  "table_all": ['coffeetable_1', 'snacktable_1', 'studytable_1'],
                  "stovefire_all": ['stovefire_1', 'stovefire_2', 'stovefire_3', 'stovefire_4'],
                  "couch_all": ['loveseat_1', 'loveseat_2', 'armchair_1', 'armchair_2', 'armchair_3', 'armchair_4'],
                  "cup_all": ['mug_1', 'longcup_1', 'longcup_2', 'glass_1'],
                  "syrup_all": ['syrup_1', 'syrup_2'],
                  "shelf_all": ['shelf_1', 'shelf_2'],
                  "book_all": ['book_1', 'book_2', 'book_3'],
                  "garbage_all": ['garbagebag_1', 'garbagebin_1'],
                  "pillow_all": ['pillow_1', 'pillow_4', 'pillow_3', 'pillow_2'],
                  "stoveknob_all": ['stoveknob_2', 'stoveknob_1', 'stoveknob_3', 'stoveknob_4', 'stoveknob'],
                  "plate_all": ['plate_1', 'plate_2'],
                  "channelall": ['channel1', 'channel2', 'channel3', 'channel4'],
                  "cd_all": ['cd_1', 'cd_2']}

#This function is used in SJI calculation
#Used to replace object instances e.g. Stovefire_1 will be replaced by StoveFire_All
#This is used for constraints which are of form (state pred1 pred2)
#return value is of form state(pred1, pred2) to be similar as planner final state output
def generalise2(action_seq):
    for i in range(len(action_seq)):
        act = action_seq[i].lstrip().replace("[","").replace("(","").replace("]","").replace(")","")
        pred1 = act.split()[1]
        pred2 = act.split()[2]
        obj_str = [pred1, pred2]
        obj_str_gen = []
        #obj_str = act[act.index("(")+1:act.index(")")].split(",")
        for obj in obj_str:
            found = False
            for key in instance_table.keys():
                # vals = [val.lower() for val in instance_table[key]]
                if obj.lower() in instance_table[key]:
                    #action += key + ","
                    # print(key)
                    obj_str_gen.append(key)
                    found = True
                    break
            if not found:
                obj_str_gen.append(obj)
        state = act.split(" ")[0].lower()
        constr = state+"("+obj_str_gen[0].lower()+","+obj_str_gen[1].lower()+")"
        action_seq[i] = constr
    return action_seq


#This function is used in IED calculation
#Used to replace object instances e.g. Stovefire_1 will be replaced by StoveFire_All
#This is used for actions which are of form moveto(plate_1)
#return value is of form moveto(plate_1) 
def generalise(action_seq):
    for i in range(len(action_seq)):
        act = action_seq[i]
        action = act[:act.index("(")] + "("
        obj_str = act[act.index("(")+1:act.index(")")].split(",")
        for obj in obj_str:
            found = False
            for key in instance_table.keys():
                # vals = [val.lower() for val in instance_table[key]]
                if obj.lower() in instance_table[key]:
                    action += key + ","
                    # print(key)
                    found = True
                    break
            if not found:
                action += obj.lower() + ","
        action = action[:-1] + ")"
        action_seq[i] = action
    return action_seq

#  add(icecreamscoop(mug_1)) --->  add(icecreamscoop, mug_1)
# ["moveto(book_1)", "keep(book_1 on coffeetable_1)"]  ----> ["moveto(book_All)"", "keep(book_All,on,Table_All)"]
def generalise3(action_seq):
    # print("At gen: ", action_seq)
    for i in range(len(action_seq)):
        act = action_seq[i]
        action = act[:act.index("(")] + "("
        # objects = act[act.index("(")+1:].replace("(", ",").replace(")", "") + ")"
        # action += objects
        obj_str = act[act.index("(")+1 : act.index(")")].split()
        for obj in obj_str:
            found = False
            for key in instance_table.keys():
                # vals = [val.lower() for val in instance_table[key]]
                if obj.lower() in instance_table[key]:
                    action += key + ","
                    # print(key)
                    found = True
                    break
            if not found:
                action += obj.lower() + ","
        action = action[:-1] + ")"
        action_seq[i] = action
    return action_seq

# l = ['moveto(plate_1)', 'grasp(plate_1)']
# print(generalise(l))
# subprocess.call("./run.sh")

#used in SJI calcualtion to make all states lower case
def preprocess3(action_list):
    ans = []
    for act in action_list:

        tmp = act.lower()
        if tmp=="":
            continue
        ans.append(tmp+")")
    return ans

#used in IED
def preprocess1(action_list):
    ans = []
    for act in action_list:

        tmp = act.lower().replace(" ", "")
        if tmp=="":
            continue
        ans.append(tmp+")")
    return ans

#used in IED
def preprocess2(action_list):
    ans = []
    for act in action_list:
        pre, post = act.split("(")[0], act.split("(")[1][:-1].split(",")
        if "_" in pre:
            if pre == "on_keep":
                tmp = "keep(" + post[0] + ",on," + post[1]+")"
            elif pre=="keep_on_sink":
                tmp = "keep("+ post[0] + ",on,sink)"
            else:
                words = pre.split("_")
                tmp = words[0] + "(" + words[1] + ")"
            ans.append(tmp)
        else:
            ans.append(act)
    return ans

#IED calculation
def minDis(s1, s2, n, m, dp):
    # If any string is empty,
    # return the remaining characters of other string
    if (n == 0):
        return m
    if (m == 0):
        return n

    # To check if the recursive tree
    # for given n & m has already been executed
    if (dp[n][m] != -1):
        return dp[n][m]
    # If characters are equal, execute
    # recursive function for n-1, m-1
    if (s1[n - 1] == s2[m - 1]):
        if (dp[n - 1][m - 1] == -1):
            dp[n][m] = minDis(s1, s2, n - 1, m - 1, dp)
            return dp[n][m]
        else:
            dp[n][m] = dp[n - 1][m - 1]
            return dp[n][m]

    # If characters are nt equal, we need to
    # find the minimum cost out of all 3 operations.
    else:
        if (dp[n - 1][m] != -1):
            m1 = dp[n - 1][m]
        else:
            m1 = minDis(s1, s2, n - 1, m, dp)

        if (dp[n][m - 1] != -1):
            m2 = dp[n][m - 1]
        else:
            m2 = minDis(s1, s2, n, m - 1, dp)
        if (dp[n - 1][m - 1] != -1):
            m3 = dp[n - 1][m - 1]
        else:
            m3 = minDis(s1, s2, n - 1, m - 1, dp)

        dp[n][m] = 1 + min(m1, min(m2, m3))
        return dp[n][m]

#used in SJI
def get_changed_state(data):
    words = data.split("\\n")
    initial_state = []
    final_state = []
    delta_g = []
    delta_g_inv = []
    for i in range(len(words)):
        if words[i][0:7] == "INITIAL":
            tmp = words[i].split()
            initial_state.append(tmp[-1])
        if words[i][0:5] == "FINAL":
            tmp = words[i].split()
            final_state.append(tmp[-1])
    for state in final_state:
       if state not in initial_state:
         delta_g.append(state)
    for state in initial_state:
       if state not in final_state:
          delta_g_inv.append(state)
    return delta_g+delta_g_inv
    # return delta_g

def get_steps(data):
    # words = re.split(' |\n|', data)
    words = data.split("\\n")
    # print("AFTER SPLIT \n ", words)
    constr = []
    for i in range(len(words)):
        if words[i][0:4] == "STEP":
            tmp = words[i].split()
            constr.append(tmp[-1])
    return constr

def calculate_jac_index(filename, pred_state, true_state):
   #calculate jac index for this instance
   union = len(pred_state)+len(true_state)
   inter = 0
   for s in pred_state:
      if s in true_state:
         inter+=1
         union-=1
   jac = inter *1.0/union
   clause_file = filename.split("_")[0]
   if clause_file in Score_jac.keys():
       Score_jac[clause_file].append(jac)
   else:
       Score_jac[clause_file] = [jac]
#    print("jac score = " + str(jac) )
   return jac
   #add to Score value

print("Hey this is Python Script Running\n")
print("Enter input model folder")
print("Enter pddl path")
print("Enter out result path")
file_input = sys.argv[1]  #exp folder
outfile_input = sys.argv[2] #pddl files folder
res_output = sys.argv[3] #output of this script file name
no_planner = sys.argv[4]
folder_name = "/home/cse/dual/cs5170493/scratch/Model/" + file_input

# raw_data = list(os.walk(raw_folder))
file_out = open(folder_name + "/" + res_output, "w")
files = os.listdir(folder_name + "/" + outfile_input)
total = int(len(files)/2)
acc = 0
count = 0

file_inconsistence = open(folder_name + "/" + "inconsistent.txt", "w")

print("Num files = ", len(files))
# for path, dirs, files in tqdm(files):
for i in range(int(len(files)/2)):
    # f = files[i]
    if i>174: break
    file_path = folder_name + "/" + outfile_input + "/" + str(i) + "_true.pddl"
    # ground truth pddl contains - sent, delta_g, action_seq
    true_pddl = open(file_path, 'r')
    Lines_true = true_pddl.readlines()
    dp_file = Lines_true[5].split(":")[1].strip()
    print("File= ", dp_file.split("_")[0])
    # print("DP file name = ", dp_file)
    # file_out.write("\nInstr: " + Lines_true[0])
    # file_out.write("True goal: " + Lines_true[1])
    # file_out.write("true action: " + Lines_true[2])
    # print("Line = ",Lines_true[2])

    print("raw str1 = ", Lines_true[2].strip())
    # Act_list: moveto(book_1), keep(book_1 on coffeetable_1)
    str1 = generalise3(Lines_true[2].split(":")[-1].strip().split(", "))
    # str1 = generalise3(preprocess1(Lines_true[2].rstrip().replace("'", "").split(":")[1].split("),")))
    if len(str1) > 0: count += 1
    
    delta_g_gen = generalise2(preprocess3(Lines_true[3].rstrip().replace("'", "").split(":")[1].split("),")))
    try:
      delta_g_inv_gen = generalise2(preprocess3(Lines_true[4].rstrip().replace("'", "").split(":")[1].split("),")))
    except:
      delta_g_inv_gen = []
    # #####################################################################3

    file_path = folder_name + "/" + outfile_input + "/" + str(i) + "_pred.pddl"

    out = subprocess.Popen(['./run_final_state.sh', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout2, stderr2 = out.communicate()
    print(stdout2)
    pred_pddl = open(file_path, 'r')
    Lines_pred = pred_pddl.readlines()
    
    # file_out.write("Pred goal: " + Lines_pred[-2])
    # file_out.write("pred action: ")
    # print(no_planner)
    if int(no_planner)==1:
        pred_changed_states = Lines_pred[-2][1:-1].split("(")[2][:-5].strip()
        print("PRED: ", pred_changed_states)
        pred_changed_states = generalise2([pred_changed_states])

    else:
        pred_changed_states = get_changed_state(str(stdout2))
        pred_changed_states = [state.replace("("," ").replace(","," ") for state in pred_changed_states]
        pred_changed_states = generalise2(pred_changed_states)
    # print("Instr: ", Lines[0][:-1])
    # print("True changed states: ", delta_g_gen+delta_g_inv_gen)
    # print("pred_changed_states: ", pred_changed_states)
    jac = calculate_jac_index(dp_file.split("_")[0], pred_changed_states,delta_g_gen+delta_g_inv_gen)
    # print("jac = ", jac)
    jac_list.append(jac)
# (:goal (AND (
    # dg_0 = Lines_pred[-2][:-1][12:-4].split()
    # dg_0 = dg_0[0]+"("+dg_0[1]+","+dg_0[2]+")"
    # jac = calculate_jac_index(f, dg_0, delta_g_gen+delta_g_inv_gen)
    print("raw str2 = ", get_steps(str(stdout2)))
    str2 = generalise(preprocess2(get_steps(str(stdout2))))
    # print("file - ", i)
    # if len(str1)==0:
    print("str1: ", str1)
    print("str2: ", str2)
    # if stderr2 is None:
    #     # for st in str2:
    #     #     file_out.write(str(st) + " ")
    # else:
    #     print("stderr pred: ", stderr2)

    n = len(str1)
    m = len(str2)
    dp = [[-1 for i in range(m + 1)] for j in range(n + 1)]

    # print("n = ", n, " m = ", m)
    if n > 0 and m > 0:
        distance = minDis(str1, str2, n, m, dp)
        tmp = (1 - (distance / max(n, m)))
        #distance_new = min(n, m) - (max(n, m) - distance)
        #tmp = (1 - (distance_new / min(n, m)))
        acc += tmp    
    else:
        distance = max(n, m)
        tmp = 0
    # print(tmp)
    clause_file = dp_file.split("_")[0]
    print("IED = ", tmp)
    ied_list.append(tmp)
    if clause_file in Score.keys():
        Score[clause_file].append(tmp)
    else:
        Score[clause_file] = [tmp]

        
    # file_out.write("SJI = " + str(jac) + "\n")
    # file_out.write("IED = " + str(tmp) + "\n")
    # inconsistent sji and ied
    # if tmp>0.4 and jac<0.4:
    dg_str, action_pred , dg_pred, dg_true= "", "", "", ""
    for dg in delta_g_gen+delta_g_inv_gen:
        dg_str += dg+" "
    if stderr2 is None:
        for st in str2:
            action_pred += st+" "
    for d in pred_changed_states:
        dg_pred += d+" "
    file_out.write(dp_file)
    file_out.write("\nInstr: " + Lines_true[0])
    file_out.write("True dg: " + dg_str)
    file_out.write("\nPred dg[0]: " + Lines_pred[-2][:-1])
    file_out.write("\nPred dg: " + dg_pred)
    file_out.write("sji = "+str(jac)+"\n")
    file_out.write("True action: " + Lines_true[2])
    file_out.write("Pred action: " + action_pred)
    file_out.write("\nied = "+str(tmp)+"\n")

    # print("Edit dist = ", distance)
    # print("SJI = ", jac)
    # file_out.write("\nEdit distance = " + str(distance))
    # print("Edit distance = ", distance)
    # print("\n \n")


# print("Accuracy: ", acc)
# print("Avg Score: ", acc / count)
acc_total = 0
count_inst = 0
for k in Score.keys():
    print(k, " = ", Score[k])
    acc_total += np.mean(Score[k])
    count_inst += len(Score[k])
acc_total /= len(Score.keys())
# acc_total /= count_inst
print("Avg IED Total old = ", acc_total)
print("Avg IED Total new = ", np.mean(ied_list))
# file_out.write("\n\nAccuracy: " + str(acc))
file_out.write("Avg IED Score: " + str(acc_total))

acc_total, count_inst = 0, 0
for k in Score_jac.keys():
    print(k, " = ",Score_jac[k])
    acc_total += np.mean(Score_jac[k])
    count_inst += len(Score[k])
# acc_total /= count_inst
acc_total /= len(Score_jac.keys())
print("Avg SJI Total old = ", acc_total)
print("Avg SJI Total new = ", np.mean(jac_list))
# file_out.write("\n\nAccuracy: " + str(acc))
file_out.write("Avg SJI Score: " + str(acc_total))
file_out.close()
file_inconsistence.close()
import pickle
import dgl
import numpy as np
import os
from os import path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as line
from .constants import *

def get_env_objects(objects):
    inter1 = len(set(objects).intersection(all_objects_kitchen))
    inter2 = len(set(objects).intersection(all_objects_living))
    if inter1 > inter2:
        objects = all_objects_kitchen
    else:
        objects = all_objects_living
    return objects

def create_pddl_gold(sent, y_true, action_seq, file_name, delta_g, delta_g_inv, dp_file_name):
    f = open(file_name + ".pddl", "w")
    f.write("sent: " + sent + "\n")
    f.write('True: ' + y_true + '\n')
    f.write("Act_list: ")
    for act in action_seq:
        if "(" in act: 
            f.write(act + ", ")
            continue
        words = act.split()
        if len(words) > 1:
            tmp = words[0] + "("
            for i in range(1, len(words)):
                tmp += words[i] + ","
            tmp = tmp[:-1] + ")"
            f.write(tmp + ", ")
    f.write("\nDelta_g: " + str(delta_g)+"\n")
    f.write("Delta_g_inv: "+str(delta_g_inv)+"\n")
    f.write("File_name: " + dp_file_name + "\n")
    f.close()


def create_pddl(init, objects, goal_list, file_name):
    f = open(file_name + ".pddl", "w")
    f.write("(define \n(problem tmp) \n(:objects ")
    for obj in objects:
        f.write(obj.lower() + " ")

    f.write(") \n(:init ")
    for state in init:
        state = state.lower()[1:-1].split()
        if len(state) > 0:
            f.write("(" + state[0] + " " + state[1])
            if len(state) > 2:
                f.write(" " + state[2] + ") ")
            else:
                f.write(") ")
    goal_string = ""
    for goal in goal_list:
        goal = goal.lower()[1:-1].split()
        if len(goal) > 0:
            goal_string += "(" + goal[0] + " " + goal[1] + " " + goal[2] + ") "
    # print("Goal string = ", goal_string)
    f.write(") \n(:goal (AND " + goal_string + ")) \n)")

    f.close()

def ind2string(ind):
    rel = all_relations[ind[0]]
    obj1 = all_objects[ind[1]]
    obj2, state = "", ""
    if ind[0] == 0 and ind[3] < N_fluents: #state preducate
        state = all_fluents[ind[3]]
        return "(" + rel + " " + obj1  + " " + state + ")" 
    if ind[2] < N_objects:
        obj2 = all_objects[ind[2]]
    return "(" + rel + " " + obj1  + " " + obj2 + ")"   


# n_relations + n_objects + (n_objects + n_fluents)
def vect2index(vect):
    v_rel = vect[0:N_relations]
    action_index = v_rel.index(max(v_rel))
    v_obj1 = vect[N_relations: N_relations + N_objects]
    obj1_index = v_obj1.index(max(v_obj1))
    v_obj2 = vect[N_relations + N_objects: N_relations + N_objects + N_objects + 1]
    obj2_index = v_obj2.index(max(v_obj2))
    v_state = vect[N_relations + N_objects + N_objects + 1:]
    state_index = v_state.index(max(v_state))
    return [action_index, obj1_index, obj2_index, state_index]


def vect2string(vect):
    ind = vect2index(vect)
    rel = all_relations[ind[0]]
    obj1 = all_objects[ind[1]]
    out = ""
    if ind[0]==0 and ind[3]<N_fluents:
        state = all_fluents[ind[3]]
        out = "(" + rel + " " + obj1  + " " + state + ")" 
    elif ind[2]<N_objects:
        obj2 = all_objects[ind[2]]
        out = "(" + rel + " " + obj1  + " " + obj2 + ")"   

    return out


def string2index(str_constr, train=True):
    words = str_constr[1:-1].split()
    if len(words) < 3:
        return [-1, -1, -1, -1]
    try:
        action_index = all_relations.index(remove_braces(words[0]))

        if words[1][-3:] == "All":
            obj1_index = all_objects.index(instance_table[words[1]][0]) if train else instance_table[words[1]]
        else:
            obj1_index = all_objects.index(words[1])

        if words[0].lower() == "state":
            obj2_index = N_objects  # None
            if words[2][-3:] == "All":
                state_index = all_fluents.index(instance_table[words[2]][0]) if train else instance_table[words[2]]
            else:
                state_index = all_fluents.index(words[2])
        else:
            state_index = N_fluents
            if words[2][-3:] == "All":
                obj2_index = all_objects.index(instance_table[words[2]][0]) if train else instance_table[words[2]]
            else:
                obj2_index = all_objects.index(words[2])

        return [action_index, obj1_index, obj2_index, state_index]
    except:
        print("Error: ", str_constr)
        return [-1, -1, -1, -1]

# matches strings all in lowercase with object indexes in all_object list
def string2index_new(str_constr, train=True):
    words = str_constr[1:-1].split()
    if len(words) < 3:
        return [-1, -1, -1, -1]
    try:
        all_objects_lower = [obj.lower() for obj in all_objects]
        all_relations_lower = [rel.lower() for rel in all_relations]
        all_fluents_lower = [fluents.lower() for fluents in all_fluents]

        action_index = all_relations_lower.index(remove_braces(words[0]))
        obj1_index = all_objects_lower.index(words[1])

        if words[0] == "state":
            obj2_index = N_objects  # None
            state_index = all_fluents_lower.index(words[2])
        else:
            state_index = N_fluents
            obj2_index = all_objects_lower.index(words[2])

        return [action_index, obj1_index, obj2_index, state_index]

    except:
        print("Error: ", str_constr)
        return [-1, -1, -1, -1]


def string2vec(constr, lower=False):
    if not lower:
        action_index, obj1_index, obj2_index, state_index = string2index(constr)
    else:
        action_index, obj1_index, obj2_index, state_index = string2index_new(constr)

    vect = np.zeros(N_relations + (N_objects * 2) + N_fluents + 2)
    words = constr[1:-1].split()
    vect[action_index] = 1
    vect[N_relations + obj1_index] = 1
    vect[N_relations + N_objects + obj2_index] = 1
    vect[N_relations + (N_objects * 2) + 1 + state_index] = 1
    return vect


# y_pred is vector, y_test is string
def accuracy(y_pred, y_true):
    y1 = vect2index(y_pred)
    if y_true == "":
        return [0, 0, 0, 0]

    y2 = string2index(y_true, False)
    match_overall = 0
    match_rel = 0
    match_obj1 = 0
    match_obj2 = 0

    if y1[1] < N_objects and type(y2[1]) == list and all_objects[y1[1]] in y2[1]:
        y2[1] = y1[1]
    if y1[2] < N_objects and type(y2[2]) == list and all_objects[y1[2]] in y2[2]:
        y2[2] = y1[2]
    if y1[3] < N_fluents and type(y2[3]) == list and all_fluents[y1[3]] in y2[3]:
        y2[3] = y1[3]

    if y1[0] == 0: #relation = state
        if y1[0:2] == y2[0:2] and y1[3] == y2[3]:
            match_overall = 1
    else:
        if y1[0:3] == y2[0:3]:
            match_overall = 1

    if (y1[0] == y2[0]):
        match_rel= 1
    if (y1[1] == y2[1]):
        match_obj1 = 1

    if (y1[0] == 0):
        if (y1[3] == y2[3]):
            match_obj2 = 1
    else:
        if (y1[2] == y2[2]):
            match_obj2 = 1

    # all_objects.index("StoveFire_4") = 6, ("StoveFire_3") = 33 , ("StoveFire_2") = 54, ("StoveFire_1") = 21
    # all_fluents.index("StoveFire4") = 23, "StoveFire3"= 7, "StoveFire2" = 17, "StoveFire1" = 21
    if match_rel and match_obj1 and y1[0] == 2 and y_true[1:-1].split()[2][0:9].lower() == "stovefire" and y1[2] in [6, 21, 33, 54]:
        match_overall = 1
        match_obj2 = 1
    if match_rel and match_obj1 and y1[0] == 0 and y_true[1:-1].split()[2][0:9].lower() == "stovefire" and y1[3] in [7, 17, 21, 23]:
        match_overall = 1
        match_obj2 = 1
    # print("predict: ", y1)
    # print("gold: ", y2)
    return [match_overall, match_rel, match_obj1, match_obj2]


# y_pred is vector, y_test is list of constraints in string
def accuracy_lenient(y_pred, y_true):
    max_match = np.array([0, 0, 0, 0])
    y_match = y_true[0]
    for y in y_true:
        v = np.asarray(accuracy(y_pred, y))
        if np.sum(v) > np.sum(max_match):
            max_match = v
            y_match = y
    return max_match.tolist(), y_match


# pred1_obj, pred2_obj, pred2_state,
def loss_function(action, pred1_obj, pred2_obj, pred2_state, y_true, l, epoch):
    l_sum = l(action.cpu().squeeze(), y_true[: N_relations])
    l_sum += l(pred1_obj.cpu().squeeze(), y_true[N_relations: N_relations + N_objects])
    l3 = l(pred2_obj.cpu().squeeze(), y_true[N_relations + N_objects: N_relations + N_objects + N_objects + 1])
    l4 = l(pred2_state.cpu().squeeze(), y_true[N_relations + N_objects + N_objects + 1:])
    if random.random() < (50 / (epoch + 5)): # teacher forcing
        if y_true[0] == 1: #state
            l_sum += l4
        else:
            l_sum += l3
    else:
        if action.cpu().squeeze()[0] == 1: #state
            l_sum += l4
        else:
            l_sum += l3
    return l_sum

def loss_function_CE(action, pred1_obj, pred2_obj, pred2_state, y_true, l):
    
    # y_true[:N_relations].detach().cpu().numpy().argmax()
    # print("action size = ", action.cpu().size())
    # print("Y true size = ", target.size())
    target = torch.ones(1, dtype=torch.long) * y_true[:N_relations].detach().cpu().numpy().argmax()
    l_sum = l(action.cpu(),target)/N_relations

    target = torch.ones(1, dtype=torch.long) * y_true[N_relations: N_relations + N_objects].detach().cpu().numpy().argmax()
    l_sum += l(pred1_obj.cpu(), target)/N_objects

    target = torch.ones(1, dtype=torch.long) * y_true[N_relations + N_objects: N_relations + N_objects + N_objects + 1].detach().cpu().numpy().argmax()
    l_sum += l(pred2_obj.cpu(), target)/(N_objects+1)

    target = torch.ones(1, dtype=torch.long) * y_true[N_relations + N_objects + N_objects + 1:].detach().cpu().numpy().argmax()
    l_sum += l(pred2_state.cpu(), target)/N_fluents
    return l_sum


def calculate_jac_index(pred_state, true_state):
   #calculate jac index for this instance
   true_state = [s[1:-1].lower() for s in true_state]
   union = len(pred_state)+len(true_state)
   inter = 0
   for s in pred_state:
      if s in true_state:
         inter+=1
         union-=1
   jac = inter *1.0/union
#    print("jac score = " + str(jac) )
   return jac

def match_score(p, t):
    count = 0
    for i in range(4):
        if p[i] == t[i]: count+=1
    return count

def eval_accuracy(model_name, test_set, model, data_folder="", num_epoch=0):
    acc = np.array([0, 0, 0, 0])
    sji = 0
    acc_exact = np.array([0, 0, 0, 0])
    loss = 0
    l = nn.BCELoss()
    # l = nn.CrossEntropyLoss()
    epoch = 1000
    # val_out = open("val_out.txt", "w")
    for iter_num, graph in tqdm(list(enumerate(test_set.graphs)), ncols=80):
        lang_embed = test_set.lang[iter_num]
        lang_str = test_set.sents[iter_num]
        y_true_list = test_set.delta_g[iter_num]
        len_y_true_list = len(y_true_list)
        # y_true_list = [item for sublist in y_true_list for item in sublist]
        y_true = y_true_list[0]
        goalObjectsVec = torch.from_numpy(np.array(test_set.goalObjectsVec[iter_num]))
        if (y_true == ""):
            continue
        if (model_name == "baseline" or model_name == "baseline_withoutGraph"):
            y_pred = model(graph, lang_embed, False, torch.FloatTensor(string2vec(y_true)), epoch)
        if model_name == "GGCN_node_attn_max" or model_name == "GGCN_node_attn_adv" or model_name == "GGCN_node_attn_norm":
            y_pred = model(graph, lang_embed, goalObjectsVec, False, torch.FloatTensor(string2vec(y_true)), epoch)
        if model_name == "GGCN_node_attn_sum" or model_name == "GGCN_attn_pairwise" or model_name == "GAT_attn_pairwise":
            action, pred1_object, pred2_object, pred2_state = model(graph, lang_str, lang_embed, goalObjectsVec, False,
                                                                    torch.FloatTensor(string2vec(y_true)), epoch, test_set.objects[iter_num],test_set.obj_states[iter_num])
        y_pred = torch.cat((action, pred1_object, pred2_object, pred2_state), 1).flatten().cpu()
        acc_tmp = accuracy_lenient(y_pred.detach().numpy().tolist(), y_true_list)[0]


        if len_y_true_list == 1:
            acc_exact += acc_tmp
        acc += acc_tmp
        # loss += l(y_pred, torch.FloatTensor(string2vec(y_true)))
        loss += loss_function(action, pred1_object, pred2_object, pred2_state, torch.FloatTensor(string2vec(y_true)), l)
        # loss += loss_function_CE(action, pred1_object, pred2_object, pred2_state, torch.FloatTensor(string2vec(y_true)), l)

    test_size = max(len(test_set.graphs), 1)
    return acc/test_size, acc_exact/max(len(test_set.graphs), 1), sji/max(len(test_set.graphs), 1), loss/max(len(test_set.graphs), 1)


def confusion_matrix(l1, l2, classes):
    cf = np.zeros([classes, classes])
    for i in range(len(l1)):
        cf[l1[i], l2[i]] += 1

    for i in range(classes):
        for j in range(classes):
            print(int(cf[i, j]), "   ", end='')
        print("\n")


def replace_all(y_true_list, acc_stat, pred):
    for i in range(len(y_true_list)):
        y_true = y_true_list[i]
        words = y_true[1:-1].split()
        if words[1][-3:] == "All":
            if y_true == acc_stat[1] and acc_stat[0][1] == 1:
                words[1] = pred[1]
            else:
                words[1] = instance_table[words[1]][0]

        if words[2][-3:] == "All":
            if y_true == acc_stat[1] and (acc_stat[0][2] == 1 or acc_stat[0][3] == 1):
                words[2] = pred[2]
            else:
                words[2] = instance_table[words[2]][0]

        y_true_list[i] = "(" + words[0] + " " + words[1] + " " + words[2] + ")"

    return y_true_list


def print_final_constraints(model_name, test_set, bool_train, model, result_folder, pddl_folder='',
                            check_planner=False):
    # outfile = open(result_folder + "/outfile.txt", "w")
    outfile = open(result_folder + "/outfile.txt", "w")
    incorrect = open(result_folder + "/incorrect_test.txt", "w")
    print("Outfile created --- ", result_folder + "/outfile.txt")
    true, pred, acc_arr, sent_arr = [], [], [], []
    constr_list = []
    obj1_list = []
    obj2_list = []
    state_list = []
    state_none = 0
    obj2_none = 0
    epoch = 10000
    eval_dict = {}
    # for iter_num, graph in tqdm(list(enumerate(test_set.graphs)), ncols=80):
    for iter_num in range(len(list(enumerate(test_set.graphs)))):
        graph = test_set.graphs[iter_num]
        # print(test_set.sents[iter_num])
        lang_embed = test_set.lang[iter_num]
        sent = test_set.sents[iter_num]
        y_true_list = test_set.delta_g[iter_num]
        # print("y_true_list : ", y_true_list)
        len_y_true_list = len(y_true_list)
        # y_true_list = [item for sublist in y_true_list for item in sublist]
        y_true = y_true_list[0]
        goalObjectsVec = torch.from_numpy(np.array(test_set.goalObjectsVec[iter_num]))
        if (y_true == ""):
            continue
        if (model_name == "baseline" or model_name == "baseline_withoutGraph"):
            y_pred = model(graph, lang_embed, False, torch.FloatTensor(string2vec(y_true)), epoch)
        if model_name == "GGCN_attn" or model_name == "Hetero_GGCN_attn" or model_name == "GGCN_attn_pairwise_2attn" or model_name == "GGCN_attn_pairwise_dropout" or model_name == "GGCN_attn_pairwise_v2" \
                or model_name == "GGCN_node_attn_adv" or model_name == "GGCN_node_attn_norm":
            y_pred = model(graph, lang_embed, goalObjectsVec, False, torch.FloatTensor(string2vec(y_true)), epoch,
                           test_set.nodes_name[iter_num], outfile)
        if model_name == "GGCN_node_attn_sum" or model_name == "GGCN_attn_pairwise" or model_name == "GAT_attn_pairwise":
            # action, pred1_object, pred2_object, pred2_state = model(graph, lang_embed, goalObjectsVec, False,
            #                                                         torch.FloatTensor(string2vec(y_true)), epoch)
            action, pred1_object, pred2_object, pred2_state = model(graph, lang_embed, goalObjectsVec, False,
                                                                    torch.FloatTensor(string2vec(y_true)), epoch,test_set.objects[iter_num],test_set.obj_states[iter_num])


        y_pred = torch.cat((action, pred1_object, pred2_object, pred2_state), 1).flatten().cpu().detach().numpy().tolist()
        y_pred_index = vect2index(y_pred)
        state = all_relations[y_pred_index[0]]
        obj1 = all_objects[y_pred_index[1]]
        if (y_pred_index[0] == 0):
            if (y_pred_index[3] == len(all_fluents)):
                obj2 = "None"
            else:
                obj2 = all_fluents[y_pred_index[3]]
        else:
            if (y_pred_index[2] == N_objects):
                obj2 = "None"
            else:
                obj2 = all_objects[y_pred_index[2]]

        acc_stat = accuracy_lenient(y_pred, y_true_list)
        outfile.write(test_set.file_name[iter_num] + "\n")
        if acc_stat[0][0] == 0:
            outfile.write("  *****  " + test_set.sents[iter_num] + "\n")
            #copy this file to new folder
            incorrect.write(test_set.file_name[iter_num] + "\n")
            incorrect.write("sent: " + test_set.sents[iter_num] + "\n")
            incorrect.write("start env: " + ", ".join(test_set.init[iter_num])+ "\n")
            incorrect.write("delta_g true: " + ", ".join(test_set.delta_g[iter_num])+ "\n")
            incorrect.write("delta_g_inv true: " + ", ".join(test_set.delta_g_inv[iter_num])+ "\n")
            incorrect.write('delta_g pred: (' + state + ', ' + obj1 + ', ' + obj2 + ')\n\n')
            #dump those files in text file with initial, final info, delta G etc
            #share with Shailender to correct along with other info required for domain
        else:
            outfile.write(test_set.sents[iter_num] + "\n")

        y_true_index = string2index(acc_stat[1])
        constr_list.append([y_true_index[0], y_pred_index[0]])
        obj1_list.append([y_true_index[1], y_pred_index[1]])
        obj2_list.append([y_true_index[2], y_pred_index[2]])
        state_list.append([y_true_index[3], y_pred_index[3]])
        y_true_split = acc_stat[1][1:-1].split()
        true.append([y_true_split])
        pred.append([state, obj1, obj2])
        out_pred = '(' + state + ' ' + obj1 + ' ' + obj2 + ')'
        acc_arr.append(acc_stat[0][0])
        sent_arr.append(sent)

        action_seq = test_set.action_seq[iter_num]
        outfile.write("Action : ")
        for act in action_seq:
            outfile.write(act[:-1] + ", ")

        outfile.write("\nTrue list: ")
        for y_tr in y_true_list:
            outfile.write(y_tr + ", ")
        outfile.write('\nTrue: ' + acc_stat[1] + '\n')
        outfile.write('Predicted: (' + state + ', ' + obj1 + ', ' + obj2 + ')\n\n')

        if obj2 == "None":
            state_none += 1
        if not (y_true_split[0] == "state") and obj2 == "None":
            obj2_none += 1

        y_true_list = replace_all(y_true_list, acc_stat, [state, obj1, obj2])
        if check_planner:
            try:
                os.makedirs(pddl_folder)
            except:
                pass
            env_obj = get_env_objects(test_set.env_objects[iter_num])
            f = pddl_folder + str(iter_num) + "_true"
            create_pddl_gold(sent, acc_stat[1], test_set.action_seq[iter_num], f, test_set.delta_g[iter_num] , test_set.delta_g_inv[iter_num], test_set.file_name[iter_num])
            f = pddl_folder + str(iter_num) + "_pred"
            comma = " "
            create_pddl(test_set.init[iter_num], env_obj, [out_pred], f)

    analyse_pred(true, pred, acc_arr, sent_arr, result_folder)

    with open("output.pkl", 'wb') as file:
        pickle.dump(true, file)
        pickle.dump(pred, file)

    outfile.close()
    incorrect.close()

def analyse_pred(y_true, y_pred, acc_arr, sent_arr, result_folder):
    all_wrong = 0
    state_cor_objs_wrong = 0
    state_cor_obj1_cor_obj2_wrong = 0
    state_cor_obj1_wrong_obj2_cor = 0
    state_wrong_objs_correct = 0
    state_wrong_obj1_cor_obj2_wrong = 0
    state_wrong_obj1_wrong_obj2_cor = 0
    wrong_obj1 = 0
    wrong_obj2 = 0
    wrong_state = 0
    total = 0
    wrong_arr = []
    wrong_state_arr = []
    for i in range(len(y_true)):
        if (acc_arr[i] == 1):  # y_true[i][0] == y_pred[i]):
            continue

        words = y_true[i][0]
        state_true = words[0]
        obj1_true = words[1]
        obj2_true = words[2]

        state_pred = y_pred[i][0]
        obj1_pred = y_pred[i][1]
        obj2_pred = y_pred[i][2]

        wrong_obj1_flag = False
        if (obj1_pred != obj1_true):
            if "All" in obj1_true:
                if (obj1_true.split('_')[0] != obj1_pred.split('_')[0]):
                    wrong_obj1 += 1
                    wrong_obj1_flag = True
            else:
                wrong_obj1 += 1
                wrong_obj1_flag = True

        wrong_obj2_flag = False
        if (obj2_pred != obj2_true):
            if "All" in obj2_true:
                if (obj2_true.split('_')[0] != obj2_pred.split('_')[0]):
                    wrong_obj2 += 1
                    wrong_obj2_flag = True
            else:
                wrong_obj2 += 1
                wrong_obj2_flag = True

        if (state_true != state_pred and wrong_obj1_flag and wrong_obj2_flag):
            all_wrong += 1
        if (state_true == state_pred and wrong_obj1_flag and wrong_obj2_flag):
            state_cor_objs_wrong += 1
        if (state_true == state_pred and wrong_obj1_flag == False and wrong_obj2_flag):
            state_cor_obj1_cor_obj2_wrong += 1
            wrong_arr.append([y_true[i][0], y_pred[i], sent_arr[i]])
        if (state_true == state_pred and wrong_obj1_flag and wrong_obj2_flag == False):
            state_cor_obj1_wrong_obj2_cor += 1
        if (state_true != state_pred and wrong_obj1_flag == False and wrong_obj2_flag == False):
            state_wrong_objs_correct += 1
        if (state_true != state_pred and wrong_obj1_flag == False and wrong_obj2_flag):
            state_wrong_obj1_cor_obj2_wrong += 1
        if (state_true != state_pred and wrong_obj1_flag and wrong_obj2_flag == False):
            state_wrong_obj1_wrong_obj2_cor += 1

        if (state_pred != state_true):
            wrong_state += 1
            wrong_state_arr.append([y_true[i][0], y_pred[i], sent_arr[i]])
        total += 1

    pred_file = open(result_folder + "/ped_analysis.txt", "w")
    pred_file.write("wrong data                      : " + str(total) + "\n")
    pred_file.write("all_wrong                       : " + str(all_wrong) + "\n")
    pred_file.write("state_cor_objs_wrong            : " + str(state_cor_objs_wrong) + "\n")
    pred_file.write("state_cor_obj1_cor_obj2_wrong   : " + str(state_cor_obj1_cor_obj2_wrong) + "\n")
    pred_file.write("state_cor_obj1_wrong_obj2_cor   : " + str(state_cor_obj1_wrong_obj2_cor) + "\n")
    pred_file.write("state_wrong_objs_correct        : " + str(state_wrong_objs_correct) + "\n")
    pred_file.write("state_wrong_obj1_cor_obj2_wrong : " + str(state_wrong_obj1_cor_obj2_wrong) + "\n")
    pred_file.write("state_wrong_obj1_wrong_obj2_cor : " + str(state_wrong_obj1_wrong_obj2_cor) + "\n")
    pred_file.write("wrong_obj1                      : " + str(wrong_obj1) + "\n")
    pred_file.write("wrong_obj2                      : " + str(wrong_obj2) + "\n")
    pred_file.write("wrong_state                     : " + str(wrong_state) + "\n")
    pred_file.close()


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def plot_grad_flow(named_parameters, filename):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if type(p.grad) == type(None):
                print("None gradient ", n)
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="g")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="b" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([line.Line2D([0], [0], color="g", lw=4),
                line.Line2D([0], [0], color="r", lw=4),
                line.Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(result_folder + filename)
    plt.close('all')

def plot_graphs(train_acc_arr, val_acc_arr):
    fig, axs = plt.subplots(2)
    fig.suptitle('Loss and Acc')
    train_acc_arr_np = np.array(train_acc_arr)
    val_acc_arr_np = np.array(val_acc_arr)
    axs[0].plot(train_loss_arr, label='train')
    axs[0].plot(val_loss_arr, label='val')
    axs[1].plot(train_acc_arr_np[:, 0], label='train')
    axs[1].plot(val_acc_arr_np[:, 0], label='val')
    axs[0].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[1].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[0].title.set_text("Loss")
    axs[1].title.set_text("Overall acc")
    plt.savefig(result_folder + "graphs_overall.jpg")
    plt.close('all')

    fig, axs = plt.subplots(3)
    fig.suptitle('Acc. of individual pred')
    axs[0].plot(train_acc_arr_np[:, 1], label='train')
    axs[0].plot(val_acc_arr_np[:, 1], label='val')
    axs[1].plot(train_acc_arr_np[:, 2], label='train')
    axs[1].plot(val_acc_arr_np[:, 2], label='val')
    axs[2].plot(train_acc_arr_np[:, 3], label='train')
    axs[2].plot(val_acc_arr_np[:, 3], label='val')
    axs[0].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[1].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[2].legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    axs[0].title.set_text("State acc")
    axs[1].title.set_text("Obj1 acc")
    axs[2].title.set_text("Obj2 acc")
    plt.savefig(result_folder + "graphs_indiv.jpg")
    plt.close('all')
import pickle
import dgl
import numpy as np
import os
from os import path
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.lines as line
from .constants import *
from .helper import *
import subprocess

def obj_set(env_domain):
    return all_objects # all_objects_kitchen if env_domain == 'kitchen' else all_objects_living

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
    f.write(") \n(:goal (AND " + goal_string + ")) \n)")
    f.close()

def vect2string(action, pred1_object, pred2_object, pred2_state, env_domain):
    action_index = torch.argmax(action).item()
    if action_index == N_relations:
        return ''
    rel = all_relations[action_index]
    obj_mask = mask_kitchen if env_domain == 'kitchen' else mask_living
    obj1 = all_objects[torch.argmax(pred1_object * obj_mask * (mask_stateful if action_index == 0 else 1)).item()]
    obj2 = all_objects[torch.argmax(pred2_object * obj_mask).item()]
    state_mask = state_masks[obj1] if action_index == 0 else 1
    state = all_fluents[torch.argmax(pred2_state * state_mask).item()]
    return "(" + rel + " " + obj1  + " " + (state if action_index == 0 else obj2) + ")" 

def string2index(str_constr, train=True):
    words = str_constr.replace('(', '').replace(')', '').split()
    action_index = all_relations.index(words[0])

    if words[1][-3:] == "All":
        obj1_index = all_objects.index(instance_table[words[1]][0]) if train else instance_table[words[1]]
    else:
        obj1_index = all_objects.index(words[1])

    if words[0].lower() == "state":
        obj2_index = -1  # None
        if words[2][-3:] == "All":
            state_index = all_fluents.index(instance_table[words[2]][0]) if train else instance_table[words[2]]
        else:
            state_index = all_fluents.index(words[2])
    else:
        state_index = -1
        if words[2][-3:] == "All":
            obj2_index = all_objects.index(instance_table[words[2]][0]) if train else instance_table[words[2]]
        else:
            obj2_index = all_objects.index(words[2])

    return [action_index, obj1_index, obj2_index, state_index]

def string2vec(constr, lower=False):
    a_vect = torch.zeros(N_relations + 1, dtype=torch.float)
    obj1_vect = torch.zeros(N_objects, dtype=torch.float)
    obj2_vect = torch.zeros(N_objects, dtype=torch.float)
    state_vect = torch.zeros(N_fluents, dtype=torch.float)
    for delta in constr:
        action_index, obj1_index, obj2_index, state_index = string2index(delta)
        a_vect[action_index] = 1
        if obj1_index != -1: obj1_vect[obj1_index] = 1
        if obj2_index != -1: obj2_vect[obj2_index] = 1
        if state_index != -1: state_vect[state_index] = 1
    if len(constr) == 0: a_vect[N_relations] = 1
    return (a_vect, obj1_vect, obj2_vect, state_vect)

def loss_function(action, pred1_obj, pred2_obj, pred2_state, y_true, delta_g, l):
    a_vect, obj1_vect, obj2_vect, state_vect = y_true
    l_act, l_obj1, l_obj2, l_state = l(action, a_vect), l(pred1_obj, obj1_vect), l(pred2_obj, obj2_vect), l(pred2_state, state_vect)
    # l_act, l_obj1, l_obj2, l_state = \
    #     l(action.view(1,-1), torch.argmax(a_vect).view(-1)), \
    #     l(pred1_obj.view(1,-1), torch.argmax(obj1_vect).view(-1)), \
    #     l(pred2_obj.view(1,-1), torch.argmax(obj2_vect).view(-1)), \
    #     l(pred2_state.view(1,-1), torch.argmax(state_vect).view(-1))
    l_sum = l_act
    if delta_g:
        l_sum += l_obj1
        if a_vect[0] == 1: #state
            l_sum += l_state
        else:
            l_sum += l_obj2
    return l_sum

def get_ied(instseq1, instseq2):
    m = len(instseq1)
    n = len(instseq2)
    if min(m,n) == 0: return 0
    cost_matrix = {}
    for i in range(m+1):
        for j in range(n+1):
            if min(i,j) == 0:
                cost_matrix[str(i)+'_'+str(j)] = max(i,j)
                continue
            cost = 1
            try:
                if instseq1[i-1] == (instseq2[j-1]):
                    cost = 0
            except:
                raise Exception('levenshtein calculation error')
            a = cost_matrix[str(i-1)+'_'+str(j)]+1
            b = cost_matrix[str(i)+'_'+str(j-1)]+1
            c = cost_matrix[str(i-1)+'_'+str(j-1)]+cost
            cost_matrix[str(i)+'_'+str(j)] = min(a, min(b,c))
    ed = float(cost_matrix[str(m)+'_'+str(n)])
    return 1 - (ed / max(m,n))

def get_sji(state_dict, init_state_dict, true_state_dict, init_true_state_dict):
    total_delta_g = set(state_dict).difference(set(init_state_dict))
    total_delta_g_inv = set(init_state_dict).difference(set(state_dict))
    true_delta_g = set(true_state_dict).difference(set(init_true_state_dict))
    true_delta_g_inv = set(init_true_state_dict).difference(set(true_state_dict))
    num = len(total_delta_g.intersection(true_delta_g)) + len(total_delta_g_inv.intersection(true_delta_g_inv))
    den = len(total_delta_g.union(true_delta_g)) + len(total_delta_g_inv.union(true_delta_g_inv))
    return num / (den + 1e-8)

def get_god_index(state_dict, true_state_dict):
    state_dict, true_state_dict = set(state_dict), set(true_state_dict)
    satisfiability = (len(true_state_dict.difference(state_dict)) / len(true_state_dict))
    optimality = (len(state_dict.difference(true_state_dict)) / (len(state_dict) + 1e-8))
    return 1 - 0.5 * satisfiability - 0.5 * optimality

def convert(state):
    converted_state = []
    for i in range(len(state)):
        act = state[i].lstrip().replace("("," ").replace(")","").replace(',', ' ')
        rel = act.split()[0]
        isaction = rel.lower() in all_possible_actions
        pred1 = act.split()[1] if len(act.split()) > 1 else ''
        pred2 = act.split()[2] if len(act.split()) > 2 else ''
        obj_str = [pred1, pred2]
        constr = f'{rel} {pred1} {pred2}' if len(act.split()) > 2 else f'{rel} {pred1}' if len(act.split()) > 1 else rel
        constr = f'({constr})' if not isaction else constr
        converted_state.append(constr)
    return converted_state

def get_delta(data):
    words = data.split('\\n')
    initial_state = []; final_state = []
    delta_g = [];  delta_g_inv = []
    for i in range(len(words)):
        if words[i][0:7] == "INITIAL":
            initial_state.append(words[i].split()[-1])
        if words[i][0:5] == "FINAL":
            final_state.append(words[i].split()[-1])
    delta_g = list(set(final_state).difference(set(initial_state)))
    delta_g_inv = list(set(initial_state).difference(set(final_state)))
    return convert(delta_g), convert(delta_g_inv), convert(final_state)

def get_steps(data):
    words = data.split('\\n')
    constr = []
    for i in range(len(words)):
        if words[i][0:4] == "STEP":
            constr.append(words[i].split()[-1])
    return convert(constr)

def get_new_state(state, delta_g, delta_g_inv):
    new_state = state + delta_g
    for rel in delta_g_inv:
        if rel in state:
            new_state.remove(rel)
    return new_state

def get_graph_util(state):
    env, objects = get_environment(state)
    nodes = []
    for obj in all_objects_lower:
        node = {}
        node['populate'] = obj in objects
        node["id"] = all_objects_lower.index(obj)
        node["name"] = obj
        node["prop"] = all_obj_prop_lower[obj]
        node["vector"] = dense_vector(obj)
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
            fromID = all_objects.index(remove_braces(t[0]))
            toID = all_objects.index(remove_braces(t[1]))
            edges.append({'from': fromID, 'to': toID, 'relation': rel})
    for i, obj in enumerate(all_objects):
        edges.append({'from': i, 'to': i, 'relation': 'Empty'})
    return {'nodes': nodes, 'edges': edges}

def convertToDGLGraph_util(state):
    graph_data = get_graph_util(state)
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
    for i, node in enumerate(graph_data["nodes"]):
        if not node['populate']: continue
        states = node["state_var"]
        prop = node["prop"]
        node_id = node["id"]
        count = 0
        for state in states:
            node_states = [i.lower() for i in all_object_states_lower[node["name"]]]
            idx = node_states.index(state)
            node_fluents[node_id][idx] = 1   
        for state in prop:
            if len(state) > 0:
                idx = all_non_fluents.index(state)
                node_prop[node_id][idx] = 1 
        node_vectors[node_id] = torch.FloatTensor(node["vector"])
    feat_mat = torch.cat((node_vectors, node_fluents, node_prop), 1)
    g.ndata['feat'] = torch.cat((node_vectors, node_fluents, node_prop), 1)
    return g

def eval_accuracy(data, model, verbose = False):
    sji, god, ied = 0, 0, 0
    for iter_num, dp in tqdm(list(enumerate(data.dp_list)), leave=False, ncols=80):
        state = dp.states[0]; state_dict = dp.state_dict[0]
        init_state_dict = dp.state_dict[0]
        action_seq = []
        for i in range(len(dp.states)):
            action, pred1_object, pred2_object, pred2_state, l_h = model(state, dp.sent_embed, dp.goal_obj_embed, l_h if i else None)
            pred_delta = vect2string(action, pred1_object, pred2_object, pred2_state, dp.env_domain)
            if pred_delta == '':
                break
            dp_acc_i = int((pred_delta == '' and dp.delta_g[i] == []) or pred_delta in dp.delta_g[i]) 
            if dp_acc_i:
                action_seq.append(dp.action_seq[i])
                state = dp.states[i+1]; state_dict = dp.state_dict[i+1]
                continue
            if verbose: print(color.GREEN, 'File', color.ENDC, dp.file_path)
            if verbose: print(color.GREEN, 'Init state', color.ENDC, state_dict)
            if verbose: print(color.GREEN, 'Pred Delta', color.ENDC, pred_delta.lower())
            create_pddl(state_dict, obj_set(dp.env_domain), [pred_delta.lower()], './planner/eval')
            out = subprocess.Popen(['bash', './planner/run_final_state.sh', './planner/eval.pddl'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = out.communicate()
            if verbose: print(color.GREEN, 'Stdout', color.ENDC, stdout.decode("utf-8"))
            planner_action = get_steps(str(stdout))
            if verbose: print(color.GREEN, 'Action', color.ENDC, planner_action)
            planner_delta_g, planner_delta_g_inv, state_dict = get_delta(str(stdout))
            if verbose: print(color.GREEN, 'Delta_g', color.ENDC, planner_delta_g)
            if verbose: print(color.GREEN, 'Final state', color.ENDC, state_dict)
            action_seq.extend(planner_action)
            state = convertToDGLGraph_util(state_dict)
        sji += get_sji(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])
        god += get_god_index(state_dict, dp.state_dict[-1])
        ied += get_ied(action_seq, dp.action_seq)
    return sji / len(data.dp_list), god / len(data.dp_list), ied / len(data.dp_list)

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
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')

def plot_graphs(result_folder, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr):
    fig, ax = plt.subplots()
    fig.suptitle('Loss and Acc')
    ax.plot(val_loss_arr, label='Val Loss', color='blue')
    ax.plot(train_loss_arr, '--', label='Train Loss', color='lightblue')
    ax.plot(val_acc_arr, label='Val Acc', color='orange')
    ax.plot(train_acc_arr, '--', label='Train Acc', color='lightcoral')
    ax.legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    plt.savefig(result_folder + "graphs_overall.pdf")
    plt.tight_layout()
    plt.close('all')

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
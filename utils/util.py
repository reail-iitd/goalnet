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
import subprocess, random
from copy import deepcopy
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--model", action="store", dest="model", default="Simple",
                  choices=['Simple', 'GGCN', 'HAN'], help="model type")
parser.add_option("-e", "--expname", action="store", dest="expname", default="",
                  help="experiment name")
parser.add_option("-r", "--train", action="store", dest="train", default="train",
                  help="training set folder")
parser.add_option("-v", "--val", action="store", dest="val", default="val",
                  help="validation set folder")
parser.add_option("-t", "--test", action="store", dest="test", default="test",
                  help="testing set folder")
opts, args = parser.parse_args()

def obj_set(env_domain):
    return all_objects # all_objects_kitchen if env_domain == 'kitchen' else all_objects_living

def create_pddl(init, objects, goal_list, file_name, inverse=False):
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
    # non-fluent properties added to pddl
    for obj in objects:
        if obj in all_obj_prop.keys():
            for prop in all_obj_prop[obj]:
                f.write("(" + prop.lower() + " " + obj.lower() + ") ")
    goal_string = ""
    for goal in goal_list:
        goal = goal.lower()[1:-1].split()
        if len(goal) > 0:
            goal_string += "(" + goal[0] + " " + goal[1] + " " + goal[2] + ") "
    if inverse:
        f.write(") \n(:goal (AND (NOT " + goal_string + "))) \n)")
    else:
        f.write(") \n(:goal (AND " + goal_string + ")) \n)")
    f.close()

def crossval(train_data, val_data):
    val_size = len(val_data.dp_list)
    all_data = train_data.dp_list + val_data.dp_list
    random.shuffle(all_data)
    train_data, val_data = all_data[val_size:], all_data[:val_size]

def check_arg_map(arg_map):
    if len(arg_map.keys()) == 0: return False
    for arg, mapping in arg_map.items():
        for noun in mapping:
            if noun.lower() not in all_objects_lower:
                return False
    return True

def vect2string(state_dict, action, pred1_object, pred2_object, pred2_state, env_domain, arg_map = None):
    action_index = torch.argmax(action).item()
    if action_index == N_relations:
        return ''
    rel = all_relations[action_index]
    obj_mask = mask_kitchen if env_domain == 'kitchen' else mask_living
    obj1_index = torch.argmax(pred1_object * obj_mask * (mask_stateful if action_index == 0 else 1)).item()
    obj1 = all_objects[obj1_index]
    obj2_mask_temp = torch.ones(len(all_objects))
    for constr in state_dict:
        if constr == '' or constr.replace('(', '').replace(')', '').split()[0] not in all_relations: continue
        a_i, o1_i, o2_i, s_i = string2index(constr)
        if action_index == a_i and obj1_index == o1_i:
            obj2_mask_temp[o2_i] = 0
    obj2_mask = 1 if action_index == 0 else obj2_mask_temp
    obj2 = all_objects[torch.argmax(pred2_object * obj_mask * obj2_mask).item()]
    state_mask = state_masks[obj1] if action_index == 0 else 1
    state = all_fluents[torch.argmax(pred2_state * state_mask).item()]
    return "(" + rel + " " + obj1  + " " + (state if action_index == 0 else obj2) + ")" 

def string2index(str_constr, train=True):
    words = str_constr.replace('(', '').replace(')', '').split()
    action_index = all_relations_lower.index(words[0].lower())
    obj1_index = all_objects_lower.index(words[1].lower())

    if words[0].lower() == "state":
        obj2_index = -1  # None
        state_index = all_fluents_lower.index(words[2].lower())
    else:
        state_index = -1
        obj2_index = all_objects_lower.index(words[2].lower())

    return [action_index, obj1_index, obj2_index, state_index]

def string2vec(state, lower=False):
    a_vect = torch.zeros(N_relations + 1, dtype=torch.float)
    obj1_vect = torch.zeros(N_objects, dtype=torch.float)
    obj2_vect = torch.zeros(N_objects, dtype=torch.float)
    state_vect = torch.zeros(N_fluents, dtype=torch.float)
    for delta in state:
        action_index, obj1_index, obj2_index, state_index = string2index(delta)
        a_vect[action_index] = 1
        if obj1_index != -1: obj1_vect[obj1_index] = 1
        if obj2_index != -1: obj2_vect[obj2_index] = 1
        if state_index != -1: state_vect[state_index] = 1
    if len(state) == 0: a_vect[N_relations] = 1
    return (a_vect, obj1_vect, obj2_vect, state_vect)

def string2embed(str_constr):
    words = str_constr.replace('(', '').replace(')', '').split()
    a_vect = torch.zeros(N_relations + 1, dtype=torch.float)
    if len(str_constr) == 0: 
        a_vect[N_relations] = 1
        obj1 = torch.ones(PRETRAINED_VECTOR_SIZE) * -1
        obj2 = torch.ones(PRETRAINED_VECTOR_SIZE) * -1
    else:
        action_index = all_relations_lower.index(words[0].lower())
        a_vect[action_index] = 1
        obj1_string = words[1]
        obj2_state_string = words[2]
        obj1 = torch.tensor((dense_vector(obj1_string)))
        obj2 = torch.tensor((dense_vector(obj2_state_string)))
    return torch.cat((a_vect,obj1,obj2))

def loss_function(action, pred1_obj, pred2_obj, pred2_state, y_true, delta_g, l):
    a_vect, obj1_vect, obj2_vect, state_vect = y_true
    l_act, l_obj1, l_obj2, l_state = l(action, a_vect), l(pred1_obj, obj1_vect), l(pred2_obj, obj2_vect), l(pred2_state, state_vect)
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

def get_sji(state_dict, init_state_dict, true_state_dict, init_true_state_dict, verbose = False):
    total_delta_g = set(state_dict).difference(set(init_state_dict))
    total_delta_g_inv = set(init_state_dict).difference(set(state_dict))
    true_delta_g = set(true_state_dict).difference(set(init_true_state_dict))
    true_delta_g_inv = set(init_true_state_dict).difference(set(true_state_dict))
    num = len(total_delta_g.intersection(true_delta_g)) + len(total_delta_g_inv.intersection(true_delta_g_inv))
    den = len(total_delta_g.union(true_delta_g)) + len(total_delta_g_inv.union(true_delta_g_inv))
    if verbose: print(color.GREEN, 'Pred total_delta_g', color.ENDC, total_delta_g)
    if verbose: print(color.GREEN, 'Pred total_delta_g_inv', color.ENDC, total_delta_g_inv)
    if verbose: print(color.GREEN, 'GT total_delta_g', color.ENDC, true_delta_g)
    if verbose: print(color.GREEN, 'GT total_delta_g_inv', color.ENDC, true_delta_g_inv)
    return num / (den + 1e-8)

def get_fbeta(state_dict, init_state_dict, true_state_dict, init_true_state_dict, beta = 2):
    total_delta_g = set(state_dict).difference(set(init_state_dict))
    total_delta_g_inv = set(init_state_dict).difference(set(state_dict))
    true_delta_g = set(true_state_dict).difference(set(init_true_state_dict))
    true_delta_g_inv = set(init_true_state_dict).difference(set(true_state_dict))
    num = (len(true_delta_g.intersection(total_delta_g)) + len(true_delta_g_inv.intersection(total_delta_g_inv)))
    precision = num / (len(total_delta_g) + len(total_delta_g_inv) + 1e-8)
    recall = num / (len(true_delta_g) + len(true_delta_g_inv) + 1e-8)
    return (1 + beta ** 2) * precision * recall / (beta * beta * precision + recall + 1e-9)

def get_fbeta_state(state_dict, true_state_dict, beta = 2):
    state_dict, true_state_dict = set(state_dict), set(true_state_dict)
    precision = len(state_dict.intersection(true_state_dict)) / (len(state_dict) + 1e-9)
    recall = len(state_dict.intersection(true_state_dict)) / (len(true_state_dict) + 1e-9)
    return (1 + beta ** 2) * precision * recall / (beta * beta * precision + recall + 1e-9)

def get_f1_index(state_dict, init_state_dict, true_state_dict, init_true_state_dict):
    total_delta_g = set(state_dict).difference(set(init_state_dict))
    total_delta_g_inv = set(init_state_dict).difference(set(state_dict))
    true_delta_g = set(true_state_dict).difference(set(init_true_state_dict))
    true_delta_g_inv = set(init_true_state_dict).difference(set(true_state_dict))
    num = (len(true_delta_g.intersection(total_delta_g)) + len(true_delta_g_inv.intersection(total_delta_g_inv)))
    precision = num / (len(total_delta_g) + len(total_delta_g_inv) + 1e-8)
    recall = num / (len(true_delta_g) + len(true_delta_g_inv) + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)

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
            idx = all_object_states_lower[node["name"]].index(state)
            node_fluents[node_id][idx] = 1   
        for state in prop:
            if len(state) > 0:
                idx = all_non_fluents.index(state)
                node_prop[node_id][idx] = 1 
        node_vectors[node_id] = torch.FloatTensor(node["vector"])
    feat_mat = torch.cat((node_vectors, node_fluents, node_prop), 1)
    g.ndata['feat'] = torch.cat((node_vectors, node_fluents, node_prop), 1)
    return g

def run_planner_simple(state_dict, dp, pred_delta, verbose = False):
    if pred_delta == '': return None, dp.convertToDGLGraph(state_dict), state_dict
    state_dict = state_dict + [pred_delta]
    a_i, o1_i, o2_i, s_i = string2index(pred_delta)
    for constr in state_dict:
        if constr == '' or constr.replace('(', '').replace(')', '').split()[0] not in all_relations: continue
        a_i2, o1_i2, o2_i2, s_i2 = string2index(constr)
        if a_i2 == 0 and a_i == 0 and o1_i == o1_i2:
            state_dict.remove(constr)
            break
    state = dp.convertToDGLGraph(state_dict)
    return None, state, state_dict

def run_planner(state_dict, dp, pred_delta, pred_delta_inv, verbose = False):
    if verbose: print(color.GREEN, 'Pred Delta', color.ENDC, pred_delta.lower())
    state_dict_lower = [rel.lower() for rel in state_dict]
    if pred_delta != '':
        create_pddl(state_dict_lower, obj_set(dp.env_domain), [pred_delta.lower()], './planner/eval')
    else:
        create_pddl(state_dict_lower, obj_set(dp.env_domain), [pred_delta_inv.lower()], './planner/eval', inverse=True)
    out = subprocess.Popen(['bash', './planner/run_final_state.sh', './planner/eval.pddl'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    planner_action = get_steps(str(stdout))
    if verbose: print(color.GREEN, 'Action', color.ENDC, planner_action)
    planner_delta_g, planner_delta_g_inv, state_dict_new = get_delta(str(stdout))
    if verbose: print(color.GREEN, 'Delta_g', color.ENDC, planner_delta_g)
    if verbose: print(color.GREEN, 'Delta_g_inv', color.ENDC, planner_delta_g_inv)
    state_dict = state_dict_new if state_dict_new else state_dict # seg fault case
    state = convertToDGLGraph_util(state_dict)
    return planner_action, state, state_dict

def eval_accuracy(data, model, verbose = False):
    sji, f1, ied, fb, fbs = 0, 0, 0, 0, 0
    max_len = max([len(dp.states) - 1 for dp in data.dp_list])
    for iter_num, dp in tqdm(list(enumerate(data.dp_list)), leave=False, ncols=80):
        state = dp.states[0]; state_dict = dp.state_dict[0]
        init_state_dict = dp.state_dict[0]
        action_seq = []
        for i in range(len(dp.states) - 1):
            if verbose: print(color.GREEN, 'File: ', color.ENDC, dp.file_path)
            pred, l_h = model(state, dp.sent_embed, dp.goal_obj_embed, l_h if i else None)
            action, pred1_object, pred2_object, pred2_state, action_inv, pred1_object_inv, pred2_object_inv, pred2_state_inv = pred
            pred_delta = vect2string(state_dict, action, pred1_object, pred2_object, pred2_state, dp.env_domain, dp.arg_map)
            pred_delta_inv = vect2string(state_dict, action_inv, pred1_object_inv, pred2_object_inv, pred2_state_inv, dp.env_domain, dp.arg_map)
            if pred_delta == '' and pred_delta_inv == '':
                break
            dp_acc_i = int(pred_delta in dp.delta_g[i] or pred_delta_inv in dp.delta_g_inv[i]) 
            if dp_acc_i:
                action_seq.append(dp.action_seq[i])
                state = dp.states[i+1]; state_dict = dp.state_dict[i+1]
                if verbose: print(color.GREEN, 'GT action', color.ENDC, dp.action_seq[i])
                if verbose: print(color.GREEN, 'GT Delta_g', color.ENDC, dp.delta_g[i])
                if verbose: print(color.GREEN, 'GT Delta_g_inv', color.ENDC, dp.delta_g_inv[i])
                continue
            planner_action, state, state_dict = run_planner(state_dict, dp, pred_delta, pred_delta_inv, verbose=verbose)
            if verbose: print(color.GREEN, 'GT action', color.ENDC, dp.action_seq[i])
            if verbose: print(color.GREEN, 'GT Delta_g', color.ENDC, dp.delta_g[i])
            if verbose: print(color.GREEN, 'GT Delta_g_inv', color.ENDC, dp.delta_g_inv[i])
            action_seq.extend(planner_action)
        if verbose: print("SJI ------------ ", get_sji(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0], verbose=verbose))
        sji += get_sji(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0], verbose=verbose)
        f1 += get_f1_index(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])
        fb += get_fbeta(state_dict, init_state_dict, dp.state_dict[-1], dp.state_dict[0])
        fbs += get_fbeta_state(state_dict, dp.state_dict[-1])
        ied += get_ied(action_seq, dp.action_seq[:-1])
        if verbose: print(color.GREEN, 'Pred action seq ', color.ENDC, action_seq)
        if verbose: print(color.GREEN, 'True action seq ', color.ENDC, dp.action_seq)
        if verbose: print("IED ------------ ", get_ied(action_seq, dp.action_seq[:-1]))
    return sji / len(data.dp_list), f1 / len(data.dp_list), ied / len(data.dp_list), fb / len(data.dp_list), fbs / len(data.dp_list)

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

def plot_graphs(result_folder, graph_name, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr):
    fig, ax = plt.subplots()
    fig.suptitle('Loss and Acc')
    ax.plot(val_loss_arr, label='Val Loss', color='blue')
    ax.plot(train_loss_arr, '--', label='Train Loss', color='lightblue')
    ax.plot(val_acc_arr, label='Val Acc', color='orange')
    ax.plot(train_acc_arr, '--', label='Train Acc', color='lightcoral')
    ax.legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    plt.savefig(result_folder + graph_name +".pdf")
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

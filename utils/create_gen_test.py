from fileinput import filename
import os
import io
from tqdm import tqdm
import numpy as np
import json
import re

all_relations = ["state", "Near", "On", "In", "Grasping"]
all_objects_spaced = ['Pillow', 'Armchair', 'Flower', 'Cd', 'Kettle', 'IceCream Scoop', 'Cd', 'Canada Dry', 'Stove Knob', 'Armchair', 'Book', 'Xbox Controller', 'Syrup', 'Sink Knob', 'Beer', 'Tv Remote Volume Down Button', 'Salt', 'Tv Remote Mute Button', 'Fridge', 'Plate', 'Tv Remote Power Button', 'Fridge Left Door', 'Tv Table', 'Robot', 'Stove Knob', 'Energy Drink', 'Snack Table', 'Tv Remote', 'Garbage Bag', 'Book', 'Long Cup', 'Fork', 'Long Cup', 'Shelf', 'Syrup', 'Tv Channel Down Button', 'Pillow', 'Book', 'Spoon', 'Light', 'Tv Channel Up Button', 'Coffee Table', 'Fridge Right Door', 'Coke', 'Stove Fire', 'Counter', 'Boiled Egg', 'Ramen', 'Love seat', 'Tv Volume Up Button', 'Bag Of Chips', 'Tv', 'Tv Remote Volume Up Button', 'Stove', 'Instant Ramen', 'Love seat', 'Plate', 'armchair', 'Fridge Button', 'Microwave Button', 'Microwave Door', 'Study table', 'Stove Knob', 'Counter', 'Stove Knob', 'Sink', 'Tv Remote Channel Down Button', 'Bowl', 'IceCream', 'Tv Remote Channel Up Button', 'Xbox', 'Lamp', 'Pillow', 'Mug', 'Shelf', 'Glass', 'Microwave', 'Stove Fire', 'Stove Fire', 'Pillow', 'Tv Volume Down Button', 'Armchair', 'Tv Power Button', 'Stove Fire', 'Garbage Bin']
#all_objects_spaced = ['tankman','tanktop']

#load the concepnet vectors
def load_all_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

#find similar objects based on eucliedean distance
def find_similar_objects():
    all_vec = list(data.values())
    all_key = list(data.keys())

    for obj in all_objects_spaced:
        print("Object = ",obj)
        vec = np.zeros(300)
        for obj_i in obj.split():
            vec+=data[obj_i.lower()]
        vec/=len(obj.split())
        sim_vec = []
        for i in range(len(all_vec)):
            vec2 = all_vec[i]
            dis = np.linalg.norm(np.array(vec) - np.array(vec2))
            if(dis<0.8):
                sim_vec.append([all_key[i],dis])
        
        sim_vec.sort(key = lambda x: x[1])
        print(sim_vec)
    
#data = load_all_vectors("../data_clean/numberbatch-en-19.08.txt")
#find_similar_objects()

#store embeddings of new objects
def gen_obj_embed():
    file_inst = open("../data_clean/gentest_simobjects.txt", 'r').readlines()
    gen_obj_dict = {}
    for line in file_inst:
        obj_list = line.split("=")[1].rstrip().split(",")
        print(obj_list)
        for obj in obj_list:
            try:
                gen_obj_dict[obj] = data[obj]
            except:
                continue
   
    s = json.dumps(gen_obj_dict)
    s = re.sub(r',\s*"', ',\n"', s)
    s = '{\n' + s[1:-1] + '\n}' 
    print(s)

#write o/p in json file
#>>python create_gen_test.py > ../jsons/gen_obj.jsongi
#gen_obj_embed()

#replace objects
def find_objects(input):
    all_obj = []
    for ip in input:
        if(len(ip)>0):
            ip = ip[0].replace("(","")
            ip = ip.replace(")","")
            ip_arr = ip.split(" ")
            all_obj.append(ip_arr[1])
            if(ip_arr[0]!="state"):
                all_obj.append(ip_arr[2])
    all_obj = list(set(all_obj))
    return all_obj

def action_replacement(action_seq, obj, gen_obj_dict):
    new_action = action_seq
    for (i,ac) in enumerate(action_seq):
        print("old action"+ac)
        new_action[i] = re.sub(obj, gen_obj_dict[obj][0], ac, flags=re.IGNORECASE)
        print("new action = " + new_action[i])
    return new_action

def state_replacement(initial_states, obj,gen_obj_dict ):
    new_initial_states = initial_states
    for (i,initial_state_i) in enumerate(initial_states):
        for (j,state) in enumerate(initial_state_i):
            new_state = state
            state = state.replace("(","").replace(")","").split(" ")
            action = state[0]
            obj1 = state[1]
            if(action in all_relations):
                obj2 = state[2]
            #replace obj1 and obj2
            if(obj == obj1.split("_")[0].lower()):
                obj1 = obj1.lower().replace(obj,gen_obj_dict[obj][0])
            if(action in ["Near","On","In","Grasping"]):
                if(obj == obj2.split("_")[0].lower()):
                    obj2 = obj2.lower().replace(obj,gen_obj_dict[obj][0])
            if(action in all_relations):
                new_state = "("+action+" "+obj1+" "+obj2+")"
            else:
                new_state = "("+action+" "+obj1+")"
            new_initial_states[i][j] = new_state
    return new_initial_states

def replace_util(obj_list,dp,gen_obj_dict):
    new_dp = dp
    arg_mapping = dp["arg_mapping"]
    new_arg_mapping = arg_mapping
    #--------------------------------------#--------------------------------------
    #replace sentence
    for obj in obj_list:   #these are the objects manipulated, change only these
        if(obj in gen_obj_dict.keys()):  #check if object present in generalization list
            if(obj in dp["sent"]):
                new_dp["sent"] = dp["sent"].replace(obj,gen_obj_dict[obj][0])
    #--------------------------------------#--------------------------------------

    #--------------------------------------#--------------------------------------
    #replace arg mapping
            if(obj in arg_mapping.keys()):
                #replace key
                new_arg_mapping[gen_obj_dict[obj][0]] = new_arg_mapping[obj]
                del new_arg_mapping[obj]
            
            #replace values (object replacement)
            for key,value in new_arg_mapping.items():
                new_value = []
                for v in value:
                    v_check = v.lower().split("_")[0].replace("(","").replace(")","")
                    if(obj == v_check):
                        #print("replace " + obj + " with " + gen_obj_dict[obj][0] + " in " + v )
                        new_value.append(v.lower().replace(obj,gen_obj_dict[obj][0]))
                    else:
                        new_value.append(v)
                new_arg_mapping[key] = new_value
            
    #--------------------------------------#--------------------------------------
            new_initial_states = state_replacement(dp["initial_states"], obj, gen_obj_dict)
            new_delta_g = state_replacement(dp["delta_g"], obj, gen_obj_dict)
            new_delta_g_inv = state_replacement(dp["delta_g_inv"], obj, gen_obj_dict)
            new_action = action_replacement(dp["action_seq"], obj, gen_obj_dict)
    #--------------------------------------#--------------------------------------

    new_dp["arg_mapping"] = new_arg_mapping
    new_dp["initial_states"] = new_initial_states
    new_dp["delta_g"] = new_delta_g
    new_dp["delta_g_inv"] = new_delta_g_inv
    new_dp["action_seq"] = new_action
    return new_dp

def replace_objects():
    data_folder = "../data_clean/test/"
    gen_data = "../data_clean/gentest/"

    ####
    file_inst = open("../data_clean/gentest_simobjects.txt", 'r').readlines()
    gen_obj_dict = {}
    for line in file_inst:
        gen_obj_list = line.split("=")[1].rstrip().split(",")
        obj = line.split("=")[0]
        gen_obj_dict[obj] = gen_obj_list
    print(gen_obj_dict)
    ####

    all_files = os.listdir(data_folder)
    for f in all_files:
        with open(data_folder + f, "r") as fh:
            print(f)
            dp = json.load(fh)
            
            #find the objects which are in delta_g and delta_g_inv
            obj_list = []
            obj_list.extend(find_objects(dp["delta_g"]))
            obj_list.extend(find_objects(dp["delta_g_inv"]))
            obj_list = [obj.split("_")[0].lower() for obj in obj_list]
            obj_list = list(set(obj_list))
    
            #replace these objects in the datapoint
        
            for obj in obj_list:
                if obj in gen_obj_dict.keys():
                    new_dp = replace_util(obj_list,dp,gen_obj_dict) 
                    f_path = gen_data + f  
                    with open(f_path, 'w') as fh:
                            json.dump(new_dp, fh, indent=4)
                    break
            

replace_objects()
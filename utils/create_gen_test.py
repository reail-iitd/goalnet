from fileinput import filename
import os
import io
from tqdm import tqdm
import numpy as np
import json
import re

all_relations = ["state", "Near", "On", "In", "Grasping"]
all_objects_spaced = ['Pillow', 'Armchair', 'Flower', 'Cd', 'Kettle', 'IceCream Scoop', 'Cd', 'Canada Dry', 'Stove Knob', 'Armchair', 'Book', 'Xbox Controller', 'Syrup', 'Sink Knob', 'Beer', 'Tv Remote Volume Down Button', 'Salt', 'Tv Remote Mute Button', 'Fridge', 'Plate', 'Tv Remote Power Button', 'Fridge Left Door', 'Tv Table', 'Robot', 'Stove Knob', 'Energy Drink', 'Snack Table', 'Tv Remote', 'Garbage Bag', 'Book', 'Long Cup', 'Fork', 'Long Cup', 'Shelf', 'Syrup', 'Tv Channel Down Button', 'Pillow', 'Book', 'Spoon', 'Light', 'Tv Channel Up Button', 'Coffee Table', 'Fridge Right Door', 'Coke', 'Stove Fire', 'Counter', 'Boiled Egg', 'Ramen', 'Love seat', 'Tv Volume Up Button', 'Bag Of Chips', 'Tv', 'Tv Remote Volume Up Button', 'Stove', 'Instant Ramen', 'Love seat', 'Plate', 'armchair', 'Fridge Button', 'Microwave Button', 'Microwave Door', 'Study table', 'Stove Knob', 'Counter', 'Stove Knob', 'Sink', 'Tv Remote Channel Down Button', 'Bowl', 'IceCream', 'Tv Remote Channel Up Button', 'Xbox', 'Lamp', 'Pillow', 'Mug', 'Shelf', 'Glass', 'Microwave', 'Stove Fire', 'Stove Fire', 'Pillow', 'Tv Volume Down Button', 'Armchair', 'Tv Power Button', 'Stove Fire', 'Garbage Bin']
all_objects = ["IceCream_1", "Fork_1", "Microwave", "Armchair_4", "Xbox_1", "Counter_1", "GarbageBin_1", "StoveFire_4", "Book_1", "Pillow_2", "Cd_2", "InstantRamen_1", "Glass_1", "Book_2", "Kettle", "Salt_1", "Shelf_1", "Beer_1", "LongCup_1", "Pillow_4", "Plate_1", "Tv_1Remote_1", "StoveFire_1", "Loveseat_1", "Book_3", "Armchair_2", "Plate_2", "Tv_1", "SnackTable_1", "BoiledEgg_1", "Sink", "Cd_1", "Bowl_1", "Loveseat_2", "StoveFire_3", "Pillow_1", "Fridge", "Studytable_1", "CanadaDry_1", "Spoon_1", "Syrup_1", "GarbageBag_1", "Mug_1", "Pillow_3", "XboxController_1", "Coke_1", "SinkKnob", "BagOfChips_1", "Syrup_2", "Armchair_3", "EnergyDrink_1", "Stove", "LongCup_2", "Armchair_1", "CoffeeTable_1", "StoveFire_2", "Ramen_1", "Shelf_2", "Robot"]
all_obj_prop  = {"Robot":  [""], "Counter_1":  ["IsPlaceableOn"], "Sink":  ["IsPlaceableOn"], "Stove":  ["Turnable"], "Mug_1":  ["IsGraspable", "IsPourable"], "MicrowaveDoor":  [""], "Microwave":  ["Openable", "Pressable", "IsPlaceableIn"], "Fridge":  ["Pressable", "Openable", "IsPlaceableIn"], "FridgeLeftDoor":  [""], "FridgeRightDoor":  [""], "Spoon_1":  ["IsGraspable", "IsPlacable", "IsScoopable"], "IceCream_1":  ["IsGraspable"], "Kettle":  ["IsGraspable", "IsPourable"], "Ramen_1":  ["IsGraspable", "IsAddable"], "Syrup_1":  ["IsGraspable", "IsSqueezeable"], "Glass_1":  ["IsGraspable", "IsPourable"], "LongCup_1":  ["IsGraspable", "IsPourable"], "LongCup_2":  ["IsGraspable", "IsPourable"], "Fork_1":  ["IsGraspable", "IsPlacable"], "EnergyDrink_1":  ["IsGraspable", "IsPourable"], "Coke_1":  ["IsGraspable", "IsPourable"], "CanadaDry_1":  ["IsGraspable", "IsPourable"], "Plate_1":  ["IsGraspable"], "Plate_2":  ["IsGraspable"], "Syrup_2":  ["IsGraspable", "IsSqueezeable"], "InstantRamen_1":  ["IsGraspable", "IsPourable"], "BoiledEgg_1":  ["IsGraspable", "IsAddable"], "Salt_1":  ["IsGraspable", "IsAddable"], "Light_1":  [""], "StoveKnob_1":  [""], "StoveKnob_2":  [""], "StoveKnob_3":  [""], "StoveKnob_4":  [""], "StoveFire_1":  ["IsPlaceableOn"], "StoveFire_2":  ["IsPlaceableOn"], "StoveFire_3":  ["IsPlaceableOn"], "StoveFire_4":  ["IsPlaceableOn"], "FridgeButton":  [""], "MicrowaveButton":  [""], "SinkKnob":  ["IsTurnable"], "IceCreamScoop":  [""], "Bowl_1":  ["IsGraspable"], "Loveseat_1":  ["IsPlaceableOn"], "Armchair_1":  ["IsPlaceableOn"], "Armchair_2":  ["IsPlaceableOn"], "CoffeeTable_1":  ["IsPlaceableOn"], "TvTable_1":  [""], "Tv_1":  ["Pressable"], "Tv_1Remote_1":  ["IsGraspable"], "Pillow_1":  ["IsGraspable"], "Pillow_2":  ["IsGraspable"], "Pillow_3":  ["IsGraspable"], "SnackTable_1":  ["IsPlaceableOn"], "BagOfChips_1":  ["IsGraspable"], "GarbageBag_1":  ["IsGraspable", "IsPlaceableIn"], "GarbageBin_1":  ["IsGraspable", "IsPlaceableOn", "IsPlaceableIn"], "Shelf_1":  ["IsPlaceableOn"], "Shelf_2":  ["IsPlaceableOn"], "Book_1":  ["IsGraspable"], "Book_2":  ["IsGraspable"], "Beer_1":  ["IsGraspable"], "XboxController_1":  ["IsGraspable"], "Xbox_1":  ["IsGraspable"], "Cd_2":  ["IsGraspable"], "Cd_1":  ["IsGraspable"], "Armchair_3":  ["IsPlaceableOn"], "Armchair_4":  ["IsPlaceableOn"], "Pillow_4":  ["IsGraspable"], "Studytable_1":  ["IsPlaceableOn"], "Lamp_1":  [""], "Tv_1PowerButton":  [""], "Tv_1ChannelUpButton":  [""], "Tv_1ChannelDownButton":  [""], "Tv_1VolumeUpButton":  [""], "Tv_1VolumeDownButton":  [""], "Tv_1Remote_1PowerButton":  [""], "Tv_1Remote_1ChannelUpButton":  [""], "Tv_1Remote_1ChannelDownButton":  [""], "Tv_1Remote_1VolumeUpButton":  [""], "Tv_1Remote_1VolumeDownButton":  [""], "Tv_1Remote_1MuteButton":  [""], "Loveseat_2":  ["IsPlaceableOn"], "Counter1_1":  ["IsPlaceableOn"], "Book_3":  ["IsGraspable"], "Flower_1":  [""]}
all_objects_kitchen = ["StoveFire_1", "Glass_1", "IceCream_1", "StoveFire_2", "Counter_1", "Plate_1", "Fork_1", "Microwave", "Salt_1", "Bowl_1", "SinkKnob", "Ramen_1", "Stove", "Spoon_1", "CanadaDry_1", "BoiledEgg_1", "Coke_1", "Sink", "Plate_2", "StoveFire_4", "LongCup_2", "Mug_1", "Fridge", "StoveFire_3", "Kettle", "Syrup_1", "InstantRamen_1", "Syrup_2", "LongCup_1", "EnergyDrink_1", "Robot"]
all_objects_living =  ["Cd_1", "Armchair_4", "Loveseat_2", "Plate_1", "Pillow_2", "Counter_1", "Armchair_2", "Bowl_1", "Armchair_1", "CoffeeTable_1", "Book_2", "BagOfChips_1", "Pillow_1", "Cd_2", "Xbox_1", "Studytable_1", "Coke_1", "SnackTable_1", "Tv_1", "GarbageBin_1", "Armchair_3", "GarbageBag_1", "Shelf_1", "Loveseat_1", "Tv_1Remote_1", "Shelf_2", "Pillow_3", "Beer_1", "XboxController_1", "Pillow_4", "Book_1", "Book_3","Robot"]
all_object_states = {"IceCream_1": ["ScoopsLeft"], "Fork_1": ["Water"], "Microwave": ["DoorIsOpen", "MicrowaveIsOn"], "Xbox_1": ["CD"], "Cd_2": ["CD"], "InstantRamen_1": ["Water", "Coffee"], "Glass_1": ["Chocolate", "Water", "Coffee"], "Kettle": ["Water", "Ramen", "Coffee"], "LongCup_1": ["IceCream", "Chocolate", "Water", "Coffee"], "Plate_1": ["Ramen", "IceCream", "Chocolate", "Egg", "Water"], "Tv_1Remote_1": ["Channel2", "Volume", "Channel3", "IsOn", "Channel1", "Channel4"], "Plate_2": ["Ramen", "IceCream", "Chocolate", "Egg", "Water"], "Tv_1": ["Channel2", "Volume", "Channel3", "IsOn", "Channel1", "Channel4"], "Cd_1": ["CD"], "Bowl_1": ["HasChips", "Ramen", "Water", "IceCream"], "Fridge": ["RightDoorIsOpen", "LeftDoorIsOpen", "WaterDispenserIsOpen"], "Spoon_1": ["ScoopsLeft", "Water"], "Syrup_1": ["Vanilla", "Chocolate"], "Mug_1": ["Chocolate", "Water", "IceCream", "Coffee", "spoon"], "SinkKnob": ["TapIsOn"], "BagOfChips_1": ["IsOpen", "HasChips"], "Syrup_2": ["Vanilla", "Chocolate"], "Stove": ["StoveFire4", "StoveFire3", "StoveFire1", "StoveFire2"], "LongCup_2": ["IceCream", "Chocolate", "Water", "Coffee"]}

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
        for obj in obj_list:
            try:
                gen_obj_dict[obj] = data[obj]
                for i in range(4):
                    gen_obj_dict[obj+"_"+str(i+1)] = data[obj]
                obj_split = obj.split("_")
                for ob in obj_split:
                    gen_obj_dict[ob] = data[ob]
            except:
                continue
   
    s = json.dumps(gen_obj_dict)
    s = re.sub(r',\s*"', ',\n"', s)
    s = '{\n' + s[1:-1] + '\n}' 
    print(s)

#write o/p in json file
#>>python create_gen_test.py > ../jsons/gen_obj.json
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
    
    for obj in obj_list:   #these are the objects manipulated, change only these
        if(obj in gen_obj_dict.keys()):  #check if object present in generalization list
            
            #replace sentence
            '''
            if(obj in dp["sent"]):
                new_dp["sent"] = dp["sent"].replace(obj,gen_obj_dict[obj][0])
            '''
            #--------------------------------------#--------------------------------------
            #replace arg mapping
            '''
            if(obj in arg_mapping.keys()):
                #replace key
                new_arg_mapping[gen_obj_dict[obj][0]] = new_arg_mapping[obj]
                del new_arg_mapping[obj]
            '''
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
    gen_data = "../data_clean/gentest_3/"

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
            #obj_list = []
            #obj_list.extend(find_objects(dp["delta_g"]))
            #obj_list.extend(find_objects(dp["delta_g_inv"]))
            obj_list = all_objects
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
            

#replace_objects()

def count_instances(obj):
    count = 0
    match_obj = ""
    for o in all_objects:
        if(obj == o.lower().split("_")[0]):
            count+=1
            match_obj = o
    return count, match_obj

def make_universal_object_set():
    universal_objects = all_objects
    universal_objects_prop = all_obj_prop
    universal_objects_kitchen = all_objects_kitchen
    universal_objects_living = all_objects_living
    universal_objects_states = all_object_states
    #print(universal_objects_prop)
    print(all_objects)
    print(len(all_objects))

    print(all_objects_kitchen)
    print(len(all_objects_kitchen))

    print(all_objects_living)
    print(len(all_objects_living))

    file_inst = open("../data_clean/gentest_simobjects.txt", 'r').readlines()
    replace_dict = {}
    for line in file_inst:
        obj_replace = line.split("=")[1].rstrip().split(",")[0]
        og_obj = line.split("=")[0]
        replace_dict[og_obj] = obj_replace
    
    for i,obj in enumerate(universal_objects):
        obj_split = obj.split("_")[0].lower()
        try:
            obj_instance = obj.split("_")[1]
        except:
            obj_instance = ""
        if(obj_split in replace_dict.keys()):
            universal_objects[i] = replace_dict[obj_split] 
            if(obj_instance):
                universal_objects[i] += "_" + obj_instance

    
    for i,obj in enumerate(universal_objects_kitchen):
        obj_split = obj.split("_")[0].lower()
        try:
            obj_instance = obj.split("_")[1]
        except:
            obj_instance = ""
        if(obj_split in replace_dict.keys()):
            universal_objects_kitchen[i] = replace_dict[obj_split] 
            if(obj_instance):
                universal_objects_kitchen[i] += "_" + obj_instance

    
    for i,obj in enumerate(universal_objects_living):
        obj_split = obj.split("_")[0].lower()
        try:
            obj_instance = obj.split("_")[1]
        except:
            obj_instance = ""
        if(obj_split in replace_dict.keys()):
            universal_objects_living[i] = replace_dict[obj_split] 
            if(obj_instance):
                universal_objects_living[i] += "_" + obj_instance

    print(universal_objects)
    print(len(universal_objects))
    print(universal_objects_kitchen)
    print(len(universal_objects_kitchen))
    print(universal_objects_living)
    print(len(universal_objects_living))
make_universal_object_set()



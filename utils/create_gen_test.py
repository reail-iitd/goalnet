from fileinput import filename
import os
import io
from tqdm import tqdm
import numpy as np
import json
import re

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
    
data = load_all_vectors("../data_clean/numberbatch-en-19.08.txt")
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
gen_obj_embed()

import os
import io
from tqdm import tqdm
import numpy as np

all_objects_spaced = ['Pillow', 'Armchair', 'Flower', 'Cd', 'Kettle', 'IceCream Scoop', 'Cd', 'Canada Dry', 'Stove Knob', 'Armchair', 'Book', 'Xbox Controller', 'Syrup', 'Sink Knob', 'Beer', 'Tv Remote Volume Down Button', 'Salt', 'Tv Remote Mute Button', 'Fridge', 'Plate', 'Tv Remote Power Button', 'Fridge Left Door', 'Tv Table', 'Robot', 'Stove Knob', 'Energy Drink', 'Snack Table', 'Tv Remote', 'Garbage Bag', 'Book', 'Long Cup', 'Fork', 'Long Cup', 'Shelf', 'Syrup', 'Tv Channel Down Button', 'Pillow', 'Book', 'Spoon', 'Light', 'Tv Channel Up Button', 'Coffee Table', 'Fridge Right Door', 'Coke', 'Stove Fire', 'Counter', 'Boiled Egg', 'Ramen', 'Love seat', 'Tv Volume Up Button', 'Bag Of Chips', 'Tv', 'Tv Remote Volume Up Button', 'Stove', 'Instant Ramen', 'Love seat', 'Plate', 'armchair', 'Fridge Button', 'Microwave Button', 'Microwave Door', 'Study table', 'Stove Knob', 'Counter', 'Stove Knob', 'Sink', 'Tv Remote Channel Down Button', 'Bowl', 'IceCream', 'Tv Remote Channel Up Button', 'Xbox', 'Lamp', 'Pillow', 'Mug', 'Shelf', 'Glass', 'Microwave', 'Stove Fire', 'Stove Fire', 'Pillow', 'Tv Volume Down Button', 'Armchair', 'Tv Power Button', 'Stove Fire', 'Garbage Bin']
#all_objects_spaced = ['tankman','tanktop']

def load_all_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


data = load_all_vectors("../data_clean/numberbatch-en-19.08.txt")

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
    

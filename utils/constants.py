import pickle
import os

# LOADING DATASET SPECIFIC CONSTANTS
# couch - Loveseat, cd2 - Far Cry Game, Book_1 - Guiness Book
# objects + fluent attributes = all_objects
with open('./data/constants.json', 'r') as fh:
        constants_dict = json.load(fh)
        all_objects = constants_dict['all_objects']
        all_objects_kitchen = constants_dict['all_objects_kitchen']
        all_objects_living = constants_dict['all_objects_living']
        all_obj_prop = constants_dict['all_obj_prop']
        all_objects_spaced = constants_dict['all_objects_spaced']
        all_fluents = constants_dict['all_fluents']
        all_non_fluents = constants_dict['all_non_fluents']
        all_states = constants_dict['all_states']
MAX_REL = max([len(all_object_states[obj]) for obj in all_object_states.keys()])
N_STATES = len(all_states)
N_objects = len(all_objects)
N_relations = len(all_relations)
N_fluents = len(all_fluents)

# LOADING DATASET SPECIFIC VOCABULARY
with open('./data/vocab.json', 'r') as fh:
        VOCAB_VECT = json.load(fh)
        for k in VOCAB_VECT:
                VOCAB_VECT[k] = np.array(VOCAB_VECT[k])

# LOADING DATASET AGNOSTIC CONCEPTNET EMBEDDINGS
with open('./jsons/conceptnet.json', 'r') as fh:
        conceptnet_vectors = json.load(fh)

# MODEL CONSTANTS
PRETRAINED_VECTOR_SIZE = 300
GRAPH_HIDDEN = 64
size, layers = (4, 2)

instance_table =  {"Ramen_All": ['Ramen_1','InstantRamen_1'],
                  "StoveFireAll": ['StoveFire1', 'StoveFire2', 'StoveFire3', 'StoveFire4'],
                  "Table_All" : ['CoffeeTable_1', 'SnackTable_1' ,'Studytable_1'],
                  "StoveFire_All":  ['Beer_1', 'Coke_1', 'CanadaDry_1', 'EnergyDrink_1'], 
                  "StoveFire_All":  ['StoveFire_1', 'StoveFire_2', 'StoveFire_3', 'StoveFire_4'],
                  "Loveseat_All" : ['Loveseat_1', 'Loveseat_2'],
                  "Armchair_All" : ['Armchair_1', 'Armchair_2', 'Armchair_3', 'Armchair_4'] , 
                  "Cup_All": ['Mug_1', 'LongCup_1', 'LongCup_2', 'Glass_1'],
                  "Syrup_All": ['Syrup_1','Syrup_2'], 
                  "Shelf_All": ['Shelf_1','Shelf_2'],
                  "Book_All": ['Book_1', 'Book_2', 'Book_3'],
                  "Garbage_All" : ['GarbageBag_1','GarbageBin_1'],
                  "Pillow_All" : ['Pillow_1','Pillow_4','Pillow_3','Pillow_2'],
                  "StoveKnob_All" : ['StoveKnob_2','StoveKnob_1','StoveKnob_3','StoveKnob_4'],
                  "Plate_All" : ['Plate_1','Plate_2'], 
                  "ChannelAll" : ['Channel1', 'Channel2', 'Channel3','Channel4'],
                  "CD_All" : ['Cd_1','Cd_2']}

test_split = [2,13,20,25,40,48,54,56,58,62,70,72,\
              86,88,101,111,113,116,120,121,124,129,\
              132,138,140,143,161,167,171,177,182,186,190,\
              192,201,203,208,212,215,220,235,242,246,\
              249,254,256,257,268,269,273,274,276,278,282,\
              289,299,306,313,314,319,320,323,325,\
              340,346,354,361,362,363,379,380,385,392,405,409,\
              417,422,426,430,434,435,442,443,444,446,\
              450,454,457,460,462,465]

graspableset = ['boiledegg_1', 'ramen_1', 'salt_1', 'icecream_1', 'bagofchips_1', \
        'beer_1', 'plate_1', 'cd_2', 'xbox_1', 'plate_2', 'pillow_1', 'pillow_3',\
        'tv_1remote_1', 'xboxcontroller_1', 'book_2', 'pillow_4', 'cd_1', 'pillow_2',\
        'book_3', 'bowl_1', 'book_1', 'coke_1', 'instantramen_1', 'fork_1', 'spoon_1', 'garbagebag_1',\
        'garbagebin_1', 'kettle', 'mug_1', 'energydrink_1', 'glass_1', 'canadadry_1', 'longcup_1', 'longcup_2',\
        'syrup_1', 'syrup_2']   

all_objects_env = [['StoveFire_1', 'Glass_1', 'IceCream_1', 'StoveFire_2', 'Plate_1', 'Fork_1', 'Microwave', 'Salt_1', 'Bowl_1', 'SinkKnob', 'Ramen_1', 'Stove', 'Spoon_1', 'CanadaDry_1', 'BoiledEgg_1', 'Coke_1', 'Sink', 'Plate_2', 'StoveFire_4', 'LongCup_2', 'Mug_1', 'Fridge', 'StoveFire_3', 'Kettle', 'Syrup_1', 'InstantRamen_1', 'Syrup_2', 'LongCup_1', 'EnergyDrink_1', 'Robot']
                , ['Cd_1', 'Armchair_4', 'Loveseat_2', 'Plate_1', 'Pillow_2', 'Armchair_2', 'Bowl_1', 'Armchair_1', 'CoffeeTable_1', 'Book_2', 'BagOfChips_1', 'Pillow_1', 'Cd_2', 'Xbox_1', 'Studytable_1', 'Coke_1', 'SnackTable_1', 'Tv_1', 'GarbageBin_1', 'Armchair_3', 'GarbageBag_1', 'Shelf_1', 'Loveseat_1', 'Tv_1Remote_1', 'Shelf_2', 'Pillow_3', 'Beer_1', 'XboxController_1', 'Pillow_4', 'Book_1', 'Book_3','Robot']]


IN_obj1 = [['IceCream_1', 'Spoon_1', 'Coke_1', 'Ramen_1', 'InstantRamen_1', 'LongCup_2', 'Bowl_1', 'Kettle', 'CanadaDry_1', 'Mug_1', 'Salt_1', 'BoiledEgg_1', 'Plate_1', 'EnergyDrink_1', 'Syrup_1', 'LongCup_1', 'Glass_1']
            , ['BagOfChips_1', 'Beer_1', 'Cd_2', 'Pillow_1', 'Pillow_2', 'Book_1', 'Cd_1', 'GarbageBag_1']]

IN_obj2 = [['Kettle', 'Fridge', 'Plate_2', 'Microwave', 'Plate_1'], ['Xbox_1', 'GarbageBag_1']]

ON_obj1 = [['Syrup_2', 'IceCream_1', 'Spoon_1', 'Coke_1', 'Ramen_1', 'InstantRamen_1', 'LongCup_2', 'Bowl_1', 'Kettle', 'Fork_1', 'Mug_1', 'Plate_2', 'Salt_1', 'BoiledEgg_1', 'Plate_1', 'EnergyDrink_1', 'Syrup_1', 'LongCup_1', 'Glass_1']
            , ['Bowl_1', 'Xbox_1', 'BagOfChips_1', 'XboxController_1', 'Beer_1', 'Pillow_4', 'Cd_2', 'Pillow_3', 'Pillow_1', 'Pillow_2', 'Book_1', 'Book_3', 'GarbageBin_1', 'Tv_1Remote_1', 'Cd_1', 'GarbageBag_1', 'Book_2', 'Coke_1']]

ON_obj2 = [['StoveFire_1', 'StoveFire_4', 'Sink', 'StoveFire_3', 'StoveFire_2'], ['Armchair_3', 'Armchair_2', 'GarbageBin_1', 'SnackTable_1', 'Shelf_1', 'CoffeeTable_1', 'Studytable_1', 'Armchair_1', 'Shelf_2', 'Armchair_4', 'Loveseat_1', 'Loveseat_2']]

Grasp_obj2 = [['CanadaDry_1', 'EnergyDrink_1', 'Salt_1', 'Syrup_2', 'IceCream_1', 'Spoon_1', 'Coke_1', 'Ramen_1', 'InstantRamen_1', 'LongCup_2', 'Bowl_1', 'Kettle', 'Fork_1', 'Mug_1', 'Plate_2', 'BoiledEgg_1', 'Plate_1', 'Syrup_1', 'LongCup_1', 'Glass_1']
            , ['GarbageBag_1', 'GarbageBin_1', 'Book_3', 'Pillow_4', 'Xbox_1', 'BagOfChips_1', 'XboxController_1', 'Beer_1', 'Cd_2', 'Pillow_3', 'Pillow_1', 'Pillow_2', 'Book_1', 'Tv_1Remote_1', 'Cd_1', 'Book_2', "Bowl_1", "Coke_1"]]

state_obj1 = [['IceCream_1', 'Glass_1', 'Stove', 'Fridge', 'Microwave', 'Syrup_2', 'SinkKnob', 'Bowl_1', 'Mug_1', 'Syrup_1', 'Kettle', 'Plate_2', 'InstantRamen_1', 'Spoon_1', 'LongCup_1', 'LongCup_2', 'Plate_1'] 
                , ['BagOfChips_1', 'Cd_2', 'Tv_1', 'Plate_1', 'Xbox_1', 'Bowl_1', 'Cd_1']]
# state_obj1 = all_obj_prop.keys()
sz = N_objects*2 + N_relations + N_fluents + 2
objects = ['Kettle', 'Fridge', 'Plate_2', 'Microwave', 'Plate_1']
# print(len(set(objects).intersection(set(all_objects_kitchen)))) 

total_valid_constr0 = len(ON_obj1[0])*len(ON_obj2[0]) + len(IN_obj1[0])*len(IN_obj2[0])+ len(Grasp_obj2[0]) + len(all_objects_kitchen) + 31 #fluents of kitchen
total_valid_constr1 =  len(ON_obj1[1])*len(ON_obj2[1]) + len(IN_obj1[1])*len(IN_obj2[1]) + len(Grasp_obj2[1]) + N_objects + 7 # object fluents
total_valid_constr = [total_valid_constr0, total_valid_constr1]

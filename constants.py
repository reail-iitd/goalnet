import pickle
import os

# couch - Loveseat, cd2 - Far Cry Game, Book_1 - Guiness Book
# objects + fluent attributes = all_objects
# all_objects = ['Pillow_1', 'Armchair_2', 'Flower_1', 'Cd_1', 'Kettle', 'IceCreamScoop', 'Cd_2', 'CanadaDry_1', 'StoveKnob_2', 'Armchair_3', 'Book_2', 'XboxController_1', 'Syrup_2', 'SinkKnob', 'Beer_1', 'Tv_1Remote_1VolumeDownButton', 'Salt_1', 'Tv_1Remote_1MuteButton', 'Fridge', 'Plate_2', 'Tv_1Remote_1PowerButton', 'FridgeLeftDoor', 'TvTable_1', 'StoveKnob_4', 'EnergyDrink_1', 'SnackTable_1', 'Tv_1Remote_1', 'GarbageBag_1', 'Book_3', 'LongCup_2', 'Fork_1', 'LongCup_1', 'Shelf_1', 'Syrup_1', 'Tv_1ChannelDownButton', 'Pillow_3', 'Book_1', 'Spoon_1', 'Light_1', 'Tv_1ChannelUpButton', 'CoffeeTable_1', 'FridgeRightDoor', 'Coke_1', 'StoveFire_1', 'Counter1_1', 'BoiledEgg_1', 'Ramen_1', 'Loveseat_2', 'Tv_1VolumeUpButton', 'BagOfChips_1', 'Tv_1', 'Tv_1Remote_1VolumeUpButton', 'Stove', 'InstantRamen_1', 'Loveseat_1', 'Plate_1', 'Armchair_4', 'FridgeButton', 'MicrowaveButton', 'MicrowaveDoor', 'Studytable_1', 'StoveKnob_3', 'Counter_1', 'StoveKnob_1', 'Sink', 'Tv_1Remote_1ChannelDownButton', 'Bowl_1', 'IceCream_1', 'Tv_1Remote_1ChannelUpButton', 'Xbox_1', 'Lamp_1', 'Pillow_2', 'Mug_1', 'Shelf_2', 'Glass_1', 'Microwave', 'StoveFire_4', 'StoveFire_3', 'Pillow_4', 'Tv_1VolumeDownButton', 'Armchair_1', 'Tv_1PowerButton', 'StoveFire_2', 'GarbageBin_1', 'Robot']
all_objects = ['IceCream_1', 'Fork_1', 'Microwave', 'Armchair_4', 'Xbox_1', 'GarbageBin_1', 'StoveFire_4', 'Book_1', 'Pillow_2', 'Cd_2', 'InstantRamen_1', 'Glass_1', 'Book_2', 'Kettle', 'Salt_1', 'Shelf_1', 'Beer_1', 'LongCup_1', 'Pillow_4', 'Plate_1', 'Tv_1Remote_1', 'StoveFire_1', 'Loveseat_1', 'Book_3', 'Armchair_2', 'Plate_2', 'Tv_1', 'SnackTable_1', 'BoiledEgg_1', 'Sink', 'Cd_1', 'Bowl_1', 'Loveseat_2', 'StoveFire_3', 'Pillow_1', 'Fridge', 'Studytable_1', 'CanadaDry_1', 'Spoon_1', 'Syrup_1', 'GarbageBag_1', 'Mug_1', 'Pillow_3', 'XboxController_1', 'Coke_1', 'SinkKnob', 'BagOfChips_1', 'Syrup_2', 'Armchair_3', 'EnergyDrink_1', 'Stove', 'LongCup_2', 'Armchair_1', 'CoffeeTable_1', 'StoveFire_2', 'Ramen_1', 'Shelf_2', 'Robot']
all_objects_kitchen = ['StoveFire_1', 'Glass_1', 'IceCream_1', 'StoveFire_2', 'Plate_1', 'Fork_1', 'Microwave', 'Salt_1', 'Bowl_1', 'SinkKnob', 'Ramen_1', 'Stove', 'Spoon_1', 'CanadaDry_1', 'BoiledEgg_1', 'Coke_1', 'Sink', 'Plate_2', 'StoveFire_4', 'LongCup_2', 'Mug_1', 'Fridge', 'StoveFire_3', 'Kettle', 'Syrup_1', 'InstantRamen_1', 'Syrup_2', 'LongCup_1', 'EnergyDrink_1', 'Robot']
all_objects_living = ['Cd_1', 'Armchair_4', 'Loveseat_2', 'Plate_1', 'Pillow_2', 'Armchair_2', 'Bowl_1', 'Armchair_1', 'CoffeeTable_1', 'Book_2', 'BagOfChips_1', 'Pillow_1', 'Cd_2', 'Xbox_1', 'Studytable_1', 'Coke_1', 'SnackTable_1', 'Tv_1', 'GarbageBin_1', 'Armchair_3', 'GarbageBag_1', 'Shelf_1', 'Loveseat_1', 'Tv_1Remote_1', 'Shelf_2', 'Pillow_3', 'Beer_1', 'XboxController_1', 'Pillow_4', 'Book_1', 'Book_3','Robot']

all_obj_prop = {'Robot':  [''], 'Counter_1':  ['IsPlaceableOn'], 'Sink':  ['IsPlaceableOn'], 'Stove':  ['Turnable'], 'Mug_1':  ['IsGraspable', 'IsPourable'], 'MicrowaveDoor':  [''], 'Microwave':  ['Openable', 'Pressable', 'IsPlaceableIn'], 'Fridge':  ['Pressable', 'Openable', 'IsPlaceableIn'], 'FridgeLeftDoor':  [''], 'FridgeRightDoor':  [''], 'Spoon_1':  ['IsGraspable', 'IsPlacable', 'IsScoopable'], 'IceCream_1':  ['IsGraspable'], 'Kettle':  ['IsGraspable', 'IsPourable'], 'Ramen_1':  ['IsGraspable', 'IsAddable'], 'Syrup_1':  ['IsGraspable', 'IsSqueezeable'], 'Glass_1':  ['IsGraspable', 'IsPourable'], 'LongCup_1':  ['IsGraspable', 'IsPourable'], 'LongCup_2':  ['IsGraspable', 'IsPourable'], 'Fork_1':  ['IsGraspable', 'IsPlacable'], 'EnergyDrink_1':  ['IsGraspable', 'IsPourable'], 'Coke_1':  ['IsGraspable', 'IsPourable'], 'CanadaDry_1':  ['IsGraspable', 'IsPourable'], 'Plate_1':  ['IsGraspable'], 'Plate_2':  ['IsGraspable'], 'Syrup_2':  ['IsGraspable', 'IsSqueezeable'], 'InstantRamen_1':  ['IsGraspable', 'IsPourable'], 'BoiledEgg_1':  ['IsGraspable', 'IsAddable'], 'Salt_1':  ['IsGraspable', 'IsAddable'], 'Light_1':  [''], 'StoveKnob_1':  [''], 'StoveKnob_2':  [''], 'StoveKnob_3':  [''], 'StoveKnob_4':  [''], 'StoveFire_1':  ['IsPlaceableOn'], 'StoveFire_2':  ['IsPlaceableOn'], 'StoveFire_3':  ['IsPlaceableOn'], 'StoveFire_4':  ['IsPlaceableOn'], 'FridgeButton':  [''], 'MicrowaveButton':  [''], 'SinkKnob':  ['IsTurnable'], 'IceCreamScoop':  [''], 'Bowl_1':  ['IsGraspable'], 'Loveseat_1':  ['IsPlaceableOn'], 'Armchair_1':  ['IsPlaceableOn'], 'Armchair_2':  ['IsPlaceableOn'], 'CoffeeTable_1':  ['IsPlaceableOn'], 'TvTable_1':  [''], 'Tv_1':  ['Pressable'], 'Tv_1Remote_1':  ['IsGraspable'], 'Pillow_1':  ['IsGraspable'], 'Pillow_2':  ['IsGraspable'], 'Pillow_3':  ['IsGraspable'], 'SnackTable_1':  ['IsPlaceableOn'], 'BagOfChips_1':  ['IsGraspable'], 'GarbageBag_1':  ['IsGraspable', 'IsPlaceableIn'], 'GarbageBin_1':  ['IsGraspable', 'IsPlaceableOn', 'IsPlaceableIn'], 'Shelf_1':  ['IsPlaceableOn'], 'Shelf_2':  ['IsPlaceableOn'], 'Book_1':  ['IsGraspable'], 'Book_2':  ['IsGraspable'], 'Beer_1':  ['IsGraspable'], 'XboxController_1':  ['IsGraspable'], 'Xbox_1':  ['IsGraspable'], 'Cd_2':  ['IsGraspable'], 'Cd_1':  ['IsGraspable'], 'Armchair_3':  ['IsPlaceableOn'], 'Armchair_4':  ['IsPlaceableOn'], 'Pillow_4':  ['IsGraspable'], 'Studytable_1':  ['IsPlaceableOn'], 'Lamp_1':  [''], 'Tv_1PowerButton':  [''], 'Tv_1ChannelUpButton':  [''], 'Tv_1ChannelDownButton':  [''], 'Tv_1VolumeUpButton':  [''], 'Tv_1VolumeDownButton':  [''], 'Tv_1Remote_1PowerButton':  [''], 'Tv_1Remote_1ChannelUpButton':  [''], 'Tv_1Remote_1ChannelDownButton':  [''], 'Tv_1Remote_1VolumeUpButton':  [''], 'Tv_1Remote_1VolumeDownButton':  [''], 'Tv_1Remote_1MuteButton':  [''], 'Loveseat_2':  ['IsPlaceableOn'], 'Counter1_1':  ['IsPlaceableOn'], 'Book_3':  ['IsGraspable'], 'Flower_1':  ['']}
all_objects_spaced = ['IceCream', 'Fork', 'Microwave', 'chair', 'Xbox', 'Garbage Bin', 'Stove Fire', 'Book', 'Pillow', 'Cd', 'Instant Ramen', 'Glass', 'Book', 'Kettle', 'Salt', 'Shelf', 'Beer', 'Long Cup', 'Pillow', 'Plate', 'Tv Remote', 'Stove Fire', 'Love seat', 'Book', 'chair', 'Plate', 'Tv', 'Snack Table', 'Boiled Egg', 'Sink', 'Cd', 'Bowl', 'Love seat', 'Stove Fire', 'Pillow', 'Fridge', 'Study table', 'Canada Dry', 'Spoon', 'Syrup', 'Garbage Bag', 'Mug', 'Pillow', 'Xbox Controller', 'Coke', 'Sink Knob', 'Bag Chips', 'Syrup', 'chair', 'Energy Drink', 'Stove', 'Long Cup', 'chair', 'Coffee Table', 'Stove Fire', 'Ramen', 'Shelf', 'Robot']
# all_objects_spaced = ['Pillow', 'Armchair', 'Flower', 'Cd', 'Kettle', 'IceCream Scoop', 'Cd', 'Canada Dry', 'Stove Knob', 'Armchair', 'Book', 'Xbox Controller', 'Syrup', 'Sink Knob', 'Beer', 'Tv Remote Volume Down Button', 'Salt', 'Tv Remote Mute Button', 'Fridge', 'Plate', 'Tv Remote Power Button', 'Fridge Left Door', 'Tv Table', 'Robot', 'Stove Knob', 'Energy Drink', 'Snack Table', 'Tv Remote', 'Garbage Bag', 'Book', 'Long Cup', 'Fork', 'Long Cup', 'Shelf', 'Syrup', 'Tv Channel Down Button', 'Pillow', 'Book', 'Spoon', 'Light', 'Tv Channel Up Button', 'Coffee Table', 'Fridge Right Door', 'Coke', 'Stove Fire', 'Counter', 'Boiled Egg', 'Ramen', 'Love seat', 'Tv Volume Up Button', 'Bag Of Chips', 'Tv', 'Tv Remote Volume Up Button', 'Stove', 'Instant Ramen', 'Love seat', 'Plate', 'armchair', 'Fridge Button', 'Microwave Button', 'Microwave Door', 'Study table', 'Stove Knob', 'Counter', 'Stove Knob', 'Sink', 'Tv Remote Channel Down Button', 'Bowl', 'IceCream', 'Tv Remote Channel Up Button', 'Xbox', 'Lamp', 'Pillow', 'Mug', 'Shelf', 'Glass', 'Microwave', 'Stove Fire', 'Stove Fire', 'Pillow', 'Tv Volume Down Button', 'Armchair', 'Tv Power Button', 'Stove Fire', 'Garbage Bin']
all_fluents = ['IsOpen', 'Channel2', 'IceCream', 'Egg', 'MicrowaveIsOn', 'Ramen', 'LeftDoorIsOpen', 'StoveFire3', 'IsOn', 'Coffee', 'ScoopsLeft', 'CD', 'Water', 'Channel3', 'TapIsOn', 'HasChips', 'RightDoorIsOpen', 'StoveFire2', 'Vanilla', 'Channel4', 'Volume', 'StoveFire1', 'DoorIsOpen', 'StoveFire4', 'WaterDispenserIsOpen', 'Chocolate', 'Channel1']
all_non_fluents = ['IsTurnable', 'IsPlaceableIn', 'IsPlaceableOn', 'Pressable', 'IsSqueezeable', 'Turnable', 'IsPlacable', 'IsAddable', 'IsPourable', 'IsScoopable', 'IsGraspable', 'Openable']
all_states = ['IsPlacable', 'IsOpen', 'Channel2', 'IsPlaceableIn', 'IsAddable', 'IsSqueezeable', 'IceCream', 'Egg', 'IsGraspable', 'MicrowaveIsOn', 'Ramen', 'LeftDoorIsOpen', 'StoveFire3', 'Pressable', 'IsOn', 'Coffee', 'ScoopsLeft', 'CD', 'Water', 'Channel3', 'Openable', 'TapIsOn', 'IsPourable', 'HasChips', 'RightDoorIsOpen', 'IsTurnable', 'StoveFire2', 'Vanilla', 'Channel4', 'Volume', 'IsPlaceableOn', 'Turnable', 'StoveFire1', 'IsScoopable', 'DoorIsOpen', 'StoveFire4', 'WaterDispenserIsOpen', 'Chocolate', 'Channel1']
all_relations = ['state', 'Near', 'On', 'In', 'Grasping'] #, 'notstate', 'notnear', 'noton', 'notin']
all_object_states = {"Syrup_2"  :  {'Chocolate'},"LongCup_1"  :  {'Water', 'Coffee'},"Mug_1"  :  {'Water', 'Chocolate', 'Coffee', 'IceCream'},"Syrup_1"  :  {'Vanilla'},"IceCream_1"  :  {'ScoopsLeft'},"Kettle"  :  {'Water', 'Ramen'},"SinkKnob"  :  {'TapIsOn'},"Stove"  :  {'StoveFire2', 'StoveFire4', 'StoveFire1', 'StoveFire3'},"InstantRamen_1"  :  {'Water', 'Coffee'},"Microwave"  :  {'MicrowaveIsOn', 'DoorIsOpen'},"Tv_1"  :  {'Channel1', 'IsOn', 'Volume', 'Channel2', 'Channel3', 'Channel4'},"BagOfChips_1"  :  {'IsOpen', 'HasChips'},"Cd_1"  :  {'CD'},"Bowl_1"  :  {'HasChips'},"Cd_2"  :  {'CD'},"Fridge"  :  {'RightDoorIsOpen', 'WaterDispenserIsOpen', 'LeftDoorIsOpen'},"Plate_1" :  {'Chocolate', 'IceCream', 'Ramen', 'Egg'},"Spoon_1"  :  {'ScoopsLeft'},"Xbox_1"  :  {'CD'},"Glass_1"  :  {'Chocolate', 'Water', 'Coffee'},"LongCup_2"  :  {'Water', 'Chocolate', 'Coffee', 'IceCream'},"Plate_2"  :  {'Chocolate', 'IceCream', 'Ramen', 'Egg'}}
N_STATES = len(all_states)
N_objects = len(all_objects)
N_relations = len(all_relations)
N_fluents = len(all_fluents)
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


conceptnet_vectors = pickle.load(open("conceptnet_embeddings_new", "rb"))
infile = open("vocab_new",'rb')
VOCAB_VECT = pickle.load(infile)

# VOCAB_VECT = []
# for i in range(len(all_objects)):
# VOCAB_VECT.append(pickle.load(infile))
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
# print(sz)
# print(N_fluents)
# print(N_objects)
# count_states = 0
# for obj in all_object_states.keys():
#         print(obj)
#         count_states += len(all_obj_prop[obj])
# print(len(all_object_states.keys()))
# print(set(all_object_states).intersection(set(all_objects_living)))
# print(list(all_object_states["Bowl_1"]))
objects = ['Kettle', 'Fridge', 'Plate_2', 'Microwave', 'Plate_1']
# print(len(set(objects).intersection(set(all_objects_kitchen))))

rel_obj1_mat = [[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
 [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 
 [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

rel_obj2_mat = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
 [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0], 
 [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]]

obj1_state_mat = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# print("Bowl_1" in ON_obj1[0])
# print("Bowl_1" in ON_obj1[1])
##### On - Bowl_1, Coke_1, 

total_valid_constr0 = len(ON_obj1[0])*len(ON_obj2[0]) + len(IN_obj1[0])*len(IN_obj2[0])+ len(Grasp_obj2[0]) + len(all_objects_kitchen) + 31 #fluents of kitchen
total_valid_constr1 =  len(ON_obj1[1])*len(ON_obj2[1]) + len(IN_obj1[1])*len(IN_obj2[1]) + len(Grasp_obj2[1]) + N_objects + 7 # object fluents
total_valid_constr = [total_valid_constr0, total_valid_constr1]
# print(total_valid_constr[0] + total_valid_constr[1])
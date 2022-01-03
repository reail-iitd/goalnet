import logging
import math
import pickle
import random
import os

import json
import constant
from tester import Data, DataSegConfig
from tqdm import tqdm

all_objects_here = ['IceCream_1', 'Fork_1', 'Microwave', 'Armchair_4', 'Xbox_1', 'GarbageBin_1', 'StoveFire_4', 'Book_1', 'Pillow_2', 'Cd_2', 'InstantRamen_1', 'Glass_1', 'Book_2', 'Kettle', 'Salt_1', 'Shelf_1', 'Beer_1', 'LongCup_1', 'Pillow_4', 'Plate_1', 'Tv_1Remote_1', 'StoveFire_1', 'Loveseat_1', 'Book_3', 'Armchair_2', 'Plate_2', 'Tv_1', 'SnackTable_1', 'BoiledEgg_1', 'Sink', 'Cd_1', 'Bowl_1', 'Loveseat_2', 'StoveFire_3', 'Pillow_1', 'Fridge', 'Studytable_1', 'CanadaDry_1', 'Spoon_1', 'Syrup_1', 'GarbageBag_1', 'Mug_1', 'Pillow_3', 'XboxController_1', 'Coke_1', 'SinkKnob', 'BagOfChips_1', 'Syrup_2', 'Armchair_3', 'EnergyDrink_1', 'Stove', 'LongCup_2', 'Armchair_1', 'CoffeeTable_1', 'StoveFire_2', 'Ramen_1', 'Shelf_2', 'Robot', 'state', 'Near', 'On', 'In', 'Grasping', 'IsPlacable', 'IsOpen', 'Channel2', 'IsPlaceableIn', 'IsAddable', 'IsSqueezeable', 'IceCream', 'Egg', 'IsGraspable', 'MicrowaveIsOn', 'Ramen', 'LeftDoorIsOpen', 'StoveFire3', 'Pressable', 'IsOn', 'Coffee', 'ScoopsLeft', 'CD', 'Water', 'Channel3', 'Openable', 'TapIsOn', 'IsPourable', 'HasChips', 'RightDoorIsOpen', 'IsTurnable', 'StoveFire2', 'Vanilla', 'Channel4', 'Volume', 'IsPlaceableOn', 'Turnable', 'StoveFire1', 'IsScoopable', 'DoorIsOpen', 'StoveFire4', 'WaterDispenserIsOpen', 'Chocolate', 'Channel1']
all_objects_lower = [w.lower() for w in all_objects_here]

def pred_correct(pred, print_bool=False):
    words = pred.replace("(", "").replace(")", "").split()
    for i, w in enumerate(words):
        if w.lower() in all_objects_lower:
            words[i] = all_objects_here[all_objects_lower.index(w.lower())]
    return '(' + ' '.join(words) + ')'

def main(datatype):
    data = Data()
    
    """testing clauses from specific file, so we collect these files, and 
       generate them as the testing clauses.
    """
    data_byfile = data.getFileData()
    test_set = []
    
    test_file_id = [2,13,20,25,40,48,54,56,58,62,70,72,\
                    86,88,101,111,113,116,120,121,124,129,\
                    132,138,140,143,161,167,171,177,182,186,190,\
                    192,201,203,208,212,215,220,235,242,246,\
                    249,254,256,257,268,269,273,274,276,278,282,\
                    289,299,306,313,314,319,320,323,325,\
                    340,346,354,361,362,363,379,380,385,392,405,409,\
                    417,422,426,430,434,435,442,443,444,446,\
                    450,454,457,460,462,465]
    for k in range(len(data_byfile)):
        file_id = int(data_byfile[k].dataid.split('.')[0])
        if file_id in test_file_id:
            test_set.append(k)
    train_set = [k for k in range(len(data_byfile)) if k not in test_set]

    
    """ generate data_bycls, and separate the train-test-dev set
    """
    data_bycls = []
    for train_file_id in train_set:
        data_bycls = data_bycls + data_byfile[train_file_id].returnClauses()
    train_cls_ids = list(range(len(data_bycls)))
    for test_file_id in test_set:
        data_bycls = data_bycls + data_byfile[test_file_id].returnClauses()
    test_cls_ids = list(range(len(train_cls_ids), len(data_bycls)))
    dev_cls_ids = [train_cls_ids[k] for k in range(0,math.ceil(len(train_cls_ids) \
                                                           / constant.dev_portion))]    
    
    data_VLD = DataSegConfig(train_cls_ids, test_cls_ids, dev_cls_ids)

    results = {}
    
    constant.load_existing_kb = False
    heus = ['Heuristic1','Heuristic2','Heuristic3','Heuristic4']
    crits = ['memory_based','most_general_node','highest_node_freq','with_optimizer']#,'upper_bound']
    
    data_VLD = DataSegConfig(train_cls_ids, test_cls_ids, dev_cls_ids)
    train_data = [data_bycls[i] for i in (data_VLD.getTestSet() if datatype == 'test' else data_VLD.getDevSet() if datatype == 'val' else data_VLD.getTrainSet())]
    unique_set = set()
    counter = 0
    for dt in tqdm(train_data, ncols=80):
        if dt.sent == '': continue
        gt_inst_seq = ""
        for k in sorted(dt.instenvseq_for_evl.keys()):
            if dt.instenvseq_for_evl[k].getInst() is not None:
                gt_inst_seq += str(dt.instenvseq_for_evl[k].getInst()) + ", "

        min_id = min(dt.instenvseq_for_evl.keys())
        max_id = max(dt.instenvseq_for_evl.keys())
        pos_gt_final_env_diff = dt.instenvseq_for_evl[min_id].getEnv().getNonExistencePreds(dt.instenvseq_for_evl[max_id].getEnv().getAllPreds())
        neg_gt_final_env_diff = dt.instenvseq_for_evl[max_id].getEnv().getNonExistencePreds(dt.instenvseq_for_evl[min_id].getEnv().getAllPreds())        
        pos_gt_str = ''.join([str(pred) for pred in pos_gt_final_env_diff])
        neg_gt_str = ''.join([str(pred) for pred in neg_gt_final_env_diff])
        initial_env = ', '.join([str(pred) for pred in dt.instenvseq_for_evl[min_id].getEnv().getAllPreds()])
        
        filename = f"../{datatype}/" +  dt.fileid.split(".")[0]+"_"+str(counter)+".json"
        states = []
        for i in range(len(dt.instenvseq_for_evl)):
            states.append([pred_correct(str(pred)) for pred in dt.instenvseq_for_evl[i].getEnv().getAllPreds()])
        action = [s.replace("(", " ").replace(")", "").strip() for s in gt_inst_seq.split(",")]
        action_new = []
        for act in action:
            words = act.split()
            for i in range(1,len(words)):
                if words[i] in all_objects_lower:
                    act = act.replace(words[i], all_objects_here[all_objects_lower.index(words[i])])
            if len(act) > 0:
                action_new.append(act)
        delta_g, delta_g_inv = [], []
        for j in range(len(states)-1):
            d_g = dt.instenvseq_for_evl[j].getEnv().getNonExistencePreds(dt.instenvseq_for_evl[j+1].getEnv().getAllPreds())
            l = [pred_correct(str(pred)) for pred in d_g if len(str(pred))>0]
            delta_g.append(l) if len(l)>0 and len(l[-1])>0 else delta_g.append([])
            d_g_inv = dt.instenvseq_for_evl[j+1].getEnv().getNonExistencePreds(dt.instenvseq_for_evl[j].getEnv().getAllPreds())        
            l = [pred_correct(str(pred)) for pred in d_g_inv if len(str(pred))>0]
            delta_g_inv.append(l) if len(l)>0 and len(l[-1])>0 else delta_g_inv.append([])
        delta_g.append([]); delta_g_inv.append([])
        arg_ground = dt.getArgGroundings()
        arg_ground_new = {}
        for key in arg_ground:
            arg_ground_new[key] = []
            for l in arg_ground[key]:
                arg_ground_new[key].append(pred_correct(l.lower(), True))

        # print('\ntest on data '+dt.fileid.split(".")[0])
        # print('sent: ', dt.sent)
        # print('Groundings - ', arg_ground_new)
        # print('GT seq = ', gt_inst_seq)
        # print('GT seq = ', action_new)
        # print("Init env - ", initial_env)
        # print("GT delta_g - ", delta_g, "\nGT delta_g_inv - ", delta_g_inv) 
        dp = {
        	'sent': dt.sent,
        	'filename': dt.fileid.split(".")[0],
            'arg_mapping': arg_ground_new,
        	'initial_states': states,
        	'action_seq': action_new + ['noop'],
        	'delta_g': delta_g,
        	'delta_g_inv': delta_g_inv
        	}

        with open(filename, 'w') as fh:
            json.dump(dp, fh, indent=4)
        counter += 1

if __name__ == "__main__":
    main('train')
    main('test')
    main('val')
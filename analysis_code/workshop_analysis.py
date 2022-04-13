import numpy as np
#This code is to compute plan lenght wise avg SJI and IED for our model from HPC out file
def analyse_our_model_planlen():
    sji_ied_grr_f1_count_planlenwise = {}
    sji = 0
    ied = 0
    f1 = 0
    grr = 0
    with open("results/our_model_predictions_1003_allmetric") as fp:
        lines = fp.readlines()
        for line in lines: 
            if("True action seq" in line):
                true_action = line.split("[")[-1]
                true_action_list = true_action.split(',')
                len_true_action_list = len(true_action_list) - 1
            if("SJI" in line):
                sji = float(line.replace('\'','').split(",")[-1])
            if("IED" in line):
                ied = float(line.replace('\'','').split(",")[-1])
            if("GRR" in line):
                grr = float(line.replace('\'','').split(",")[-1])
            if("F1" in line):
                f1 = float(line.replace('\'','').split(",")[-1])
                if(len_true_action_list in sji_ied_grr_f1_count_planlenwise):
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][0] += sji
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][1] += ied
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][2] += grr
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][3] += f1
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][4] += 1
                else:
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list] = [sji,ied,grr,f1,1]
                

    for key in sorted(sji_ied_grr_f1_count_planlenwise):
        print (key, 
                sji_ied_grr_f1_count_planlenwise[key][0]/sji_ied_grr_f1_count_planlenwise[key][4],
                sji_ied_grr_f1_count_planlenwise[key][1]/sji_ied_grr_f1_count_planlenwise[key][4],
                sji_ied_grr_f1_count_planlenwise[key][2]/sji_ied_grr_f1_count_planlenwise[key][4],
                sji_ied_grr_f1_count_planlenwise[key][3]/sji_ied_grr_f1_count_planlenwise[key][4],
                sji_ied_grr_f1_count_planlenwise[key][4])


#This code is to compute constraint lenght wise avg SJI and IED for our model from HPC out file
def analyse_our_model_constraintlen():
    sji_ied_grr_f1_count_constraintlenwise = {}
    sji = 0
    ied = 0
    f1 = 0
    grr = 0
    with open("results/our_model_predictions_1003_allmetric") as fp:
        lines = fp.readlines()
        for line in lines: 
            if("GT total_delta_g  {" in line):
                true_delta_g = line.split("{")[-1]
                true_delta_g_list = true_delta_g.split(',')
                len_true_delta_g_list = len(true_delta_g_list)
            if("GT total_delta_g  set" in line):
                len_true_delta_g_list = 0

            if("GT total_delta_g_inv" in line):
                len_true_delta_g_inv_list = 0 
                true_delta_g_inv = line.split("{")
                if(len(true_delta_g_inv) > 1):
                    true_delta_g_inv = true_delta_g_inv[-1]
                    true_delta_g_inv_list = true_delta_g_inv.split(',')
                    len_true_delta_g_inv_list = len(true_delta_g_inv_list)
                len_true_constraint_list = len_true_delta_g_inv_list + len_true_delta_g_list
            if("SJI" in line):
                sji = float(line.replace('\'','').split(",")[-1])
            if("IED" in line):
                ied = float(line.replace('\'','').split(",")[-1])
            if("GRR" in line):
                grr = float(line.replace('\'','').split(",")[-1])
            if("F1" in line):
                f1 = float(line.replace('\'','').split(",")[-1])
                if(len_true_constraint_list in sji_ied_grr_f1_count_constraintlenwise):
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list][0].append(sji)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list][1].append(ied)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list][2].append(grr)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list][3].append(f1)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list][4] += 1
                else:
                    sji_ied_grr_f1_count_constraintlenwise[len_true_constraint_list] = [[sji],[ied],[grr],[f1],1]
                
    print("MEAN values")
    print("SJI  || IED || GRR || F1")
    for key in sorted(sji_ied_grr_f1_count_constraintlenwise):
        print (key, 
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][0])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][1])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][2])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][3])))
    
    print("STD values")
    print("SJI  || IED || GRR || F1")
    for key in sorted(sji_ied_grr_f1_count_constraintlenwise):
        print (key, 
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][0])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][1])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][2])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][3])))


#analyse_our_model_constraintlen()



#This code is to compute plan lenght vs constraint len for our model from HPC out file
def analyse_our_model_plan_vs_constraintlen():
    with open("results/our_model_predictions_1003_allmetric") as fp:
        lines = fp.readlines()
        for line in lines: 
            if("Pred total_delta_g  {" in line):
                pred_delta_g = line.split("{")[-1]
                pred_delta_g_list = pred_delta_g.split(',')
                len_pred_delta_g_list = len(pred_delta_g_list)
            if("Pred total_delta_g  set" in line):
                len_pred_delta_g_list = 0

            if("Pred total_delta_g_inv" in line):
                len_pred_delta_g_inv_list = 0 
                pred_delta_g_inv = line.split("{")
                if(len(pred_delta_g_inv) > 1):
                    pred_delta_g_inv = pred_delta_g_inv[-1]
                    pred_delta_g_inv_list = pred_delta_g_inv.split(',')
                    len_pred_delta_g_inv_list = len(pred_delta_g_inv_list)
                len_pred_constraint_list = len_pred_delta_g_inv_list + len_pred_delta_g_list
            if("Pred action seq" in line):
                pred_action = line.split("[")[-1]
                pred_action_list = pred_action.split(',')
                len_pred_action_list = len(pred_action_list) 
                
                print(len_pred_action_list,":",len_pred_constraint_list)
#
# analyse_our_model_plan_vs_constraintlen()



#This code is to compute plan lenght wise avg SJI and IED from old format of ACL predcitions
def analyse_acl_model_oldfile():
    sji_ied_count_planlenwise = {}
    sji = 0
    ied = 0
    file_name = "gfgf"
    with open("results/acl16_predictions.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            if("test on" in line):
                file_name = (line.split(" ")[-1]).rstrip()
                continue
            if(file_name in line):
                sji = float((line.split("  ")[-1]).rstrip())
                ied = float(line.split("  ")[-2])
            if("GT seq " in line):
                true_action = line.split("=")[-1]
                true_action_list = true_action.split(',')
                len_true_action_list = len(true_action_list) - 1
            if("GT delta_g -" in line):
                true_delta_g = line.split("-")[-1]
                len_true_delta_g_list = 0
                if(len(true_delta_g) > 5):
                    true_delta_g_list = true_delta_g.split(")(")
                    len_true_delta_g_list = len(true_delta_g_list)
                
            if("GT delta_g_inv " in line):
                true_delta_g_inv = line.split("-")[-1]
                len_true_delta_g_inv_list = 0
                if(len(true_delta_g_inv) > 5):
                    true_delta_g_inv_list = true_delta_g_inv.split(")(")
                    len_true_delta_g_inv_list = len(true_delta_g_inv_list)
                total_delta_len = len_true_delta_g_inv_list + len_true_delta_g_list#
                if(len_true_action_list in sji_ied_count_planlenwise):
                    sji_ied_count_planlenwise[len_true_action_list][0] += sji
                    sji_ied_count_planlenwise[len_true_action_list][1] += ied
                    sji_ied_count_planlenwise[len_true_action_list][2] += 1
                else:
                    sji_ied_count_planlenwise[len_true_action_list] = [sji,ied,1]
                
    total_sji = 0
    total_ied = 0
    total_count = 0
    '''
    for key in sorted(sji_ied_count_planlenwise):
        print (key, sji_ied_count_planlenwise[key][0]/sji_ied_count_planlenwise[key][2],sji_ied_count_planlenwise[key][1]/sji_ied_count_planlenwise[key][2],sji_ied_count_planlenwise[key][2])
        total_sji += sji_ied_count_planlenwise[key][0]
        total_ied += sji_ied_count_planlenwise[key][1]
        total_count += sji_ied_count_planlenwise[key][2]

    print("Avg sji = ",total_sji/total_count)
    print("Avg ied = ",total_ied/total_count)
    print("count",total_count)
    '''
#analyse_acl_model_oldfile()


#This code is to compute plan lenght wise avg SJI and IED from new format of ACL predcitions
def analyse_acl_prediction_allmetric_planlen():
    sji_ied_grr_f1_count_planlenwise = {}
    sji = 0
    ied = 0
    f1 = 0
    grr = 0
    file_planlength = []
    with open("results/acl_file_planlenght.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            file_planlength.append(int((line.split(':')[1]).rstrip()))

    count = 0
    with open("results/acl16_predictions_allmetric") as fp:
        lines = fp.readlines()
        for line in lines:
            if("IED" in line):
                ied = float((line.split(":")[-1]).rstrip())
            if("SJI" in line):
                sji = float((line.split(":")[-1]).rstrip())
            if("F1" in line):
                f1 = float((line.split(":")[-1]).rstrip())
            if("GRR" in line):
                grr = float((line.split(":")[-1]).rstrip())
                len_true_action_list = file_planlength[count]
                if(len_true_action_list == 11):
                    print("IED",ied)
                    print("SJI",sji)
                if(len_true_action_list in sji_ied_grr_f1_count_planlenwise):
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][0] += sji
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][1] += ied
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][2] += grr
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][3] += f1
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list][4] += 1
                else:
                    sji_ied_grr_f1_count_planlenwise[len_true_action_list] = [sji,ied,grr,f1,1]

                count+=1
                
    for key in sorted(sji_ied_grr_f1_count_planlenwise):
        print (key, 
            sji_ied_grr_f1_count_planlenwise[key][0]/sji_ied_grr_f1_count_planlenwise[key][4],
            sji_ied_grr_f1_count_planlenwise[key][1]/sji_ied_grr_f1_count_planlenwise[key][4],
            sji_ied_grr_f1_count_planlenwise[key][2]/sji_ied_grr_f1_count_planlenwise[key][4],
            sji_ied_grr_f1_count_planlenwise[key][3]/sji_ied_grr_f1_count_planlenwise[key][4],
            sji_ied_grr_f1_count_planlenwise[key][4])
       
#analyse_acl_prediction_allmetric()


#This code is to compute constraint length wise avg SJI and IED from new format of ACL predcitions
def analyse_acl_prediction_allmetric_constraintlen():
    sji_ied_grr_f1_count_constraintlenwise = {}
    sji = 0
    ied = 0
    f1 = 0
    grr = 0
    file_constraintlength = []
    with open("results/acl_file_constraintlen.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            file_constraintlength.append(int((line.split(':')[1]).rstrip()))

    count = 0
    with open("results/acl16_predictions_allmetric") as fp:
        lines = fp.readlines()
        for line in lines:
            if("IED" in line):
                ied = float((line.split(":")[-1]).rstrip())
            if("SJI" in line):
                sji = float((line.split(":")[-1]).rstrip())
            if("F1" in line):
                f1 = float((line.split(":")[-1]).rstrip())
            if("GRR" in line):
                grr = float((line.split(":")[-1]).rstrip())
                len_true_action_list = file_constraintlength[count]
                if(len_true_action_list in sji_ied_grr_f1_count_constraintlenwise):
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list][0].append(sji)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list][1].append(ied)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list][2].append(grr)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list][3].append(f1)
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list][4] += 1
                else:
                    sji_ied_grr_f1_count_constraintlenwise[len_true_action_list] = [[sji],[ied],[grr],[f1],1]

                count+=1
                
    print("MEAN values")
    print("SJI  || IED || GRR || F1")
    for key in sorted(sji_ied_grr_f1_count_constraintlenwise):
        print (key, 
                np.mean(np.array(
                    [key][0])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][1])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][2])),
                np.mean(np.array(sji_ied_grr_f1_count_constraintlenwise[key][3])))
    
    print("STD values")
    print("SJI  || IED || GRR || F1")
    for key in sorted(sji_ied_grr_f1_count_constraintlenwise):
        print (key, 
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][0])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][1])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][2])),
                np.std(np.array(sji_ied_grr_f1_count_constraintlenwise[key][3])))
            


#analyse_acl_prediction_allmetric_constraintlen()

#This code is to compute plan lenght wise avg SJI and IED from old format of ACL predcitions
def analyse_acl_model_oldfile_plan_vs_constraint_len():
    sji_ied_count_planlenwise = {}
    sji = 0
    ied = 0
    file_name = "gfgf"
    with open("results/acl16_predictions.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            if("Pred seq " in line):
                pred_action = line.split("=")[-1]
                pred_action_list = pred_action.split(',')
                pred_true_action_list = len(pred_action_list) - 1
            if("Pred delta_g -" in line):
                pred_delta_g = line.split("-")[-1]
                len_pred_delta_g_list = 0
                if(len(pred_delta_g) > 5):
                    pred_delta_g_list = pred_delta_g.split(")(")
                    len_pred_delta_g_list = len(pred_delta_g_list)
            if("Pred delta_g_inv -" in line):
                pred_delta_g_inv = line.split("-")[-1]
                len_pred_delta_g_inv_list = 0
                if(len(pred_delta_g_inv) > 5):
                    pred_delta_g_inv_list = pred_delta_g_inv.split(")(")
                    len_pred_delta_g_inv_list = len(pred_delta_g_inv_list)
                print(pred_true_action_list,":",len_pred_delta_g_list+len_pred_delta_g_inv_list)

#analyse_acl_model_oldfile_plan_vs_constraint_len()


#find train plan len frequency
import json
import os
def train_plan_len_freq():
    len_Action_count = {}
    folder = "data_clean/train/"
    files = os.listdir(folder)
    for f in files:
        if ('a' in f):
            continue
        file_path = folder + f
        with open(file_path, "r") as fh:
            data = json.load(fh)
        len_action = len(data["action_seq"]) - 1
        if(len_action in len_Action_count):
            len_Action_count[len_action] +=1
        else:
            len_Action_count[len_action] =1
        if(len_action > 21):
            print(data["sent"])
            #print(data["action_seq"])
    for key in sorted(len_Action_count):
        print (key,len_Action_count[key])
#train_plan_len_freq()


def check_constraint_len_5():
    with open("results/acl_file_constraintlen.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            file_constraintlength= int((line.split(':')[1]).rstrip())
            filename = line.split(':')[0]
            if(file_constraintlength == 5):
                print(filename)

#check_constraint_len_5()

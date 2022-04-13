#compare output of our model with acl predictions
count = 0
sent = ""
sentence_acl = ""
with open("results/filtered_out_2203_FCN_adjmat_2") as fp:
        lines = fp.readlines()
        for line in lines:
            if("FILE_NAME" in line):
                file_name = line.split("=")[-1]
            if("SENT" in line):
                sent = line.split("=")[-1]
            if("True action seq" in line):
                true_action = line.split("[")[-1]
                true_action_list = true_action.split(',')
                len_true_action_list = len(true_action_list) - 1
            if("GT total_delta_g {" in line):
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
                sji = float((line.split(",")[-1]).replace(')',''))
            if("IED" in line):
                ied = float((line.split(",")[-1]).replace(')',''))
                if(sji>0):
                    with open("results/acl16_predictions.txt") as fp:
                        lines = fp.readlines()
                        for line in lines:
                            if("test on" in line):
                                file_name_acl = (line.split(" ")[-1]).rstrip()
                                continue
                            if("sent" in line):
                                sentence_acl = line.split(":")[-1]
                              
                            if(file_name_acl in line):
                                sji_acl = float((line.split("  ")[-1]).rstrip())
                                ied_acl = float(line.split("  ")[-2])
                                if(sentence_acl.strip() == sent.strip()):
                                    if(len_true_constraint_list == 9):
                                        print(sentence_acl)
                                        print(file_name_acl)
                                        print("ACL score : SJI = " + str(sji_acl)+", IED = "+str(ied_acl))
                                        print("GoalNEt score : SJI = " + str(sji)+", IED = "+str(ied))
                                        print(true_action_list)


                                    
                                    

#check which example has high IED here, search for same in ACL database

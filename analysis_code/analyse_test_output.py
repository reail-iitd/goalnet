#this file reads the raw test output of hpc outfile and makes a filtered version
import os
import json
class test_output_class:
    Pred_total_delta_g = ""
    Pred_total_delta_g_inv = ""
    GT_total_delta_g = ""
    GT_total_delta_g_inv = ""
    Pred_action_seq = ""
    True_action_seq = ""
    filename = ""
    SJI = ""
    IED = ""
    sent = ""

    def print_text(self):
        print("FILE_NAME = "+self.filename)
        print("SENT = "+
        self.sent)
        print(self.Pred_total_delta_g)
        print(self.Pred_total_delta_g_inv)
        print(self.GT_total_delta_g)
        print(self.GT_total_delta_g_inv)
        print(self.Pred_action_seq)
        print(self.True_action_seq)
        print("SJI = ",self.SJI)
        print("IED = ",self.IED)


with open("results/out_2203_FCN_adjmat_2") as fp:
    lines = fp.readlines()
    test_output = test_output_class()
    for line in lines:
        if("File" in line):
           test_output.filename = (line.split(' ')[-1]).rstrip()
           file_json = json.load(open(test_output.filename))
           test_output.sent = file_json["sent"]
        if("Pred total_delta_g_inv" in line):
            test_output.Pred_total_delta_g_inv = line
            continue
        if("Pred total_delta_g" in line):
            test_output.Pred_total_delta_g = line

        if("GT total_delta_g_inv" in line):
            test_output.GT_total_delta_g_inv = line
            continue
        if("GT total_delta_g" in line):
            test_output.GT_total_delta_g = line
        
        if("Pred action seq" in line):
            test_output.Pred_action_seq = line
        if("True action seq" in line):
            test_output.True_action_seq = line
        if("IED" in line):
            test_output.IED = float((line.split(' ')[-1]).rstrip())
            #print(test_output.IED)
            if(test_output.IED <=0.1  and test_output.SJI <0.1):
                test_output.print_text()
        if("SJI" in line):
            test_output.SJI = float((line.split(' ')[-1]).rstrip())

        
        
        
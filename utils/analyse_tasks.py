import nltk
from nltk.tokenize import word_tokenize
import os
import json
def findVerb(sent):
    wordsList = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(wordsList)
    #print(tagged)
    verbs = []
    #print(tagged)
    for tag in tagged:
        if("VB" in str(tag)):
            verbs.append(tag[0])
    return verbs

'''
#find verbs in train set
verbs_set = []
all_files = os.listdir("data_clean/train/")
for f in all_files:
    with open("data_clean/train/"+f, "r") as fh:
        tmp = json.load(fh)
        sent = tmp['sent']
        verbs = findVerb(sent)
        verbs_set.extend(verbs)

verbs_set = set(verbs_set)
print(verbs_set)
'''

#lenght is 90
verbs_set_train = ['scoop', 'throw', 'toggle',  'put', 'dump', 'play', 
  'pick', 'add', 'connected', 'microwave', 'stove', 'start', 'place', 'heat', 'hook',
 'hold', 'wash', 'serve', 'grab', 'keep', 'change', 'turn', 'pour', 
'prepare', 'open', 'empty', 'insert', 'eat', 'take', 'go', 'tidy', 
'have', 'make', 'squirt', 'plug', 'fill',
 'drizzle', 'give', 'set', 'use', 'replace', 'bring', 'move', 'grasp', 
 'stack', 'boiling', 'discard', 'find', 'arrange', 'fix', 'cook', 'close', 'check', 'get', 
 'remove', 'clean', 'using', 'boiled', 'boil', 'distribute', 
 'carry', 'collect', 'connect']

verbs_set_train_not_in_test = ['drizzle', 'make', 'boiled', 'prepare', 'plug', 'dump', 
'eat', 'go', 'toggle', 'play', 'discard', 'give', 'clean', 'using', 'heat', 'carry', 
'squirt', 'connect', 'cook', 'scoop', 'place', 'stove', 'boiling', 'keep', 'insert', 
'replace', 'fix', 'remove', 'stack', 'have', 'boil', 'start', 'microwave', 'hook', 'check', 
'connected', 'tidy']

'''
tmp = ['xbox']
#read sentences from train set having above verbs
all_files = os.listdir("data_clean/train/")
for verb in tmp:
    print("VERB ==========",
    verb)
    for f in all_files:
        with open("data_clean/train/"+f, "r") as fh:
            tmp = json.load(fh)
            sent = tmp['sent']
            state_dict = tmp['initial_states'][-1]
            init_state_dict = tmp['initial_states'][0]
            total_delta_g = set(state_dict).difference(set(init_state_dict))
            total_delta_g_inv = set(init_state_dict).difference(set(state_dict))
            action_seq = tmp['action_seq'] if 'action_seq' in tmp else []
            if verb in sent and ("cd" in sent or "game" in sent):
                print(sent)
                print(total_delta_g)
                print(total_delta_g_inv)
                print(action_seq)
'''
'''
#find total verbs in the set
verbs_set = []
with open("results/filtered_output.txt") as fp:
    lines = fp.readlines()
    for line in lines:
        if("SENT" in line):
            sent = line.split("=")[-1]
            verbs = findVerb(sent)
            verbs_set.extend(verbs)

verbs_set = set(verbs_set)
print(verbs_set)
'''


#cluster sentences based on verbs
verbs_set = ['use', 'take', 'needed', 'distribute', 'arrange',
 'throw', 'pouring', 'fill', 'pour', 'collect', 'close', 'empty', 'pick',
  'bring', 'open', 'put', 'move', 'grasp', 
 'wash', 'add', 'get', 'form', 'change', 'set', 'turn', 
 'hold', 'find', 'tow', 'grab', 'serve']

verb_clusters = [['use','turn'],
                ['take', 'get', 'move','put', 'add','throw','pouring','fill','pour', 'empty'],
                ['distribute','arrange','collect','serve','set'],
                ['grab','grasp','pick','hold','bring','find'],
                ['close','open','change'],
                ['wash']]
                
count_total = 0
for i in range(6):
    vc =  verb_clusters[i]
    print(vc)
    avg_sji = 0.0
    avg_ied = 0.0
    count = 0
    with open("results/filtered_output.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            if("FILE_NAME" in line):
                file_name = line.split("=")[-1]
            if("SENT" in line):
                sent = line.split("=")[-1]
            if("SJI" in line):
                sji = float(line.split("=")[-1])
            if("IED" in line):
                ied = float(line.split("=")[-1])
                #find the cluster
                for vc_i in vc:
                    if(vc_i in sent):
                        print("FILENAME = ",file_name)
                        print("SENT = "+sent)
                        #print(ied)
                        avg_sji+=sji
                        avg_ied+=ied
                        count+=1
                        #print(sji)
    print("AVG SJI = "+str(avg_sji/count))
    print("AVG IED = "+str(avg_ied/count))
    print("Count = "+str(count))
    count_total += count
    fp.close()
print(count_total)



            

import copy
import logging

import constant

from physicalobject import PhysicalObject
from environment import Environment
from instruction import Instruction
from statepredicate import StatePredicate
from inducer import HypoNet

logger = logging.getLogger(__name__)

class instenvPair(object):
    def __init__(self, inst=None, env=None):
        self.inst = inst
        self.env = env
    def storeEnv(self, env):
        self.env = env
    def storeInst(self, inst):
        self.inst = inst
    def getEnv(self):
        return self.env
    def getInst(self):
        return self.inst
    
class ListClauseData(object):
    def __init__(self, filedir, clsfname, envfname, sml = None):
        self.dataid = clsfname
        
        self.clsf = filedir+clsfname
        self.instenvf = filedir+envfname
        self.phyobjlist = []
        self.instenvpairs = {}
        
        # this is for evaluation purpose. Cleaned the original instruction sequence 
        # to remove noisy actions or change the order of noisy action sequences
        self.instenvpairs_for_evl = {}
        
        if sml is None:
            self.sml = Simulator()
            self.sml.formSimulator()
        else:
            self.sml = sml
            
        self.clauselist = []
        
        self._extractClsList()
        
    def _extractClsList(self):
        clsfid = open(self.clsf, 'r')
        instenvfid = open(self.instenvf,'r')
        """Step1, read the instruction environment file to get:
           1. a list of physical objects
           2. the instruction sequence 
           3. the environment sequence"""
        pairid = -1
        for line in instenvfid.readlines():
            line = line.lower()
            if line.startswith('obj list:'):
                objliststr = line[line.index('('):].strip()
                for objstr in objliststr.split(' '):   
                    phyobj = PhysicalObject()
                    phyobj.buildFromStr(objstr)
                    self.phyobjlist.append(phyobj)
           
            elif line.startswith('<start step >'):
                line = line.strip()
                pairid = int(line.split('>')[1])
                self.instenvpairs[pairid] = instenvPair()
           
            elif line.startswith('env:'):
                env = Environment()
                env.createObjList(self.phyobjlist)
                env.createPredListFromDataStr(line[line.index('('):].strip())
                
                self.instenvpairs[pairid].storeEnv(env)
            
            elif line.startswith('instruction:'):
                inst = Instruction()
                instStr = line[line.index(' ')+1:].strip().lower().replace('inside','in')
                if len(instStr) != 0:
                    inst.createInstFromDataStr(instStr)
                    self.instenvpairs[pairid].storeInst(inst)
            
            elif line.startswith('<end step >'):
                continue
            elif line.startswith('\n'):
                continue
            else:
                raise Exception('instenv data file '+str(self.instenvf)+\
                                'has unidentified line!')
        instenvfid.close()
                    
        self._refine_all_states()   
#         """Step1.2, clean the ground truth action sequence to get the evaluation action sequence.
#            In the clean, remove some actions or action sequences or change the order of action sequences.
#            But the remove of these actions does not change the over all state. For example, remove wait() which
#            does not have any effect."""
#         self.instenvpairs_for_evl = \
#                 _clean_instenvpairs_to_get_seq_for_evl(self.instenvpairs)

        """Step2, create data structure for each clause."""
        start_inst_id = -1
        end_inst_id = -1
        
        evl_seq_start_id = 0
        for line in clsfid.readlines():
            line = line.lower()
            if line.startswith('start env:'):
                start_inst_id = int(line.split(':')[1].strip())
                cls = Clause()
                cls.fileid = self.dataid
            
            elif line.startswith('end env:'):
                end_inst_id = int(line.split(':')[1].strip())
                if end_inst_id < start_inst_id:
                    pass
                else:
                    cls.fillInstenvseqFromStartEnd(start_inst_id, end_inst_id, \
                                                   self.instenvpairs)
                    for i in range(evl_seq_start_id, start_inst_id):
                        self.instenvpairs_for_evl[len(self.instenvpairs_for_evl)] \
                            = copy.deepcopy(self.instenvpairs[i])
                    for k in sorted(cls.instenvseq_for_evl.keys())[:-1]:
                        self.instenvpairs_for_evl[len(self.instenvpairs_for_evl)] \
                            = copy.deepcopy(cls.instenvseq_for_evl[k])
                    evl_seq_start_id = end_inst_id+1
            
            elif line.startswith('sent:'):
                tmpsent = line.split(':')[1][1:].strip()
                cls.fillSentFromString(tmpsent)
            elif line.startswith('clause dscr:'):
                tmpdscr = line.split('clause dscr: ')[1].strip().replace(' ','').lower()
                cls.fillClsDscrFromString(tmpdscr)
            elif line.startswith('arg mapping:'):
                tmpmap = line.split(':')[1].strip().replace(' ','')
                cls.fillMapFromString(tmpmap)          
            elif line.startswith('<end clause>'):
                if start_inst_id > end_inst_id:
                    pass
                else:
                    cls.extractVerbFrame()      
                    original_heu = constant.node_chosen_heu
                    constant.node_chosen_heu = 'Heuristic1'
                    hn = HypoNet()
                    hn.createNetFromCls(cls)
                    cls.hyponet_heu1 = hn
                     
                    constant.node_chosen_heu = 'Heuristic2'
                    hn = HypoNet()
                    hn.createNetFromCls(cls)
                    cls.hyponet_heu2 = hn
                     
                    constant.node_chosen_heu = 'Heuristic3'
                    hn = HypoNet()
                    hn.createNetFromCls(cls)
                    cls.hyponet_heu3 = hn

                    constant.node_chosen_heu = 'Heuristic4'
                    hn = HypoNet()
                    hn.createNetFromCls(cls)
                    cls.hyponet_heu4 = hn
                     
                    constant.node_chosen_heu = original_heu
                     
                    self.clauselist.append(cls)
                start_inst_id = -1
                end_inst_id = -1
                                
            else:
                continue
        
        clsfid.close()
        
        for k in sorted(self.instenvpairs.keys()):
            if k >= evl_seq_start_id:
                self.instenvpairs_for_evl[len(self.instenvpairs_for_evl)] \
                    = copy.deepcopy(self.instenvpairs[k])
        """Step2.2, clean the ground truth action sequence to get the evaluation action sequence.
           In the clean, remove some actions or action sequences or change the order of action sequences.
           But the remove of these actions does not change the over all state. For example, remove wait() which
           does not have any effect."""
        self.instenvpairs_for_evl = \
                _clean_instenvpairs_to_get_seq_for_evl(self.instenvpairs_for_evl)
        #self.instenvpairs_for_evl = self.instenvpairs_for_evl


                
    def _refine_all_states(self):
        """the raw instruction action sequence data has some problem. 
           Intermediate environments are from simulator, some actions in the original
           simulator cannot corrected predict the output based on the PDDL. Here
           we refine these environments based on a more accurate simulator.
        """
        print("Running the simulator to clean states!!")
        for i in range(len(self.instenvpairs)-1):
            current_env = self.instenvpairs[i].getEnv()
            current_inst = self.instenvpairs[i].getInst()
            next_env = self.sml.execute(current_inst, current_env)
            self.instenvpairs[i+1].storeEnv(next_env)
    
    def returnClauses(self):
        return self.clauselist
    
    def getBeginningEnv(self):
        return self.instenvpairs[min(self.instenvpairs.keys())].getEnv()
    
    def getInstenvpair_for_evl(self):
        return self.instenvpairs_for_evl
           
def _clean_instenvpairs_to_get_seq_for_evl(instenvpairs):
    print("Correcting the action order!!")
    """Clean the instruction sequence by couple steps"""
    tmp_instenvpairs = copy.deepcopy(instenvpairs)
    """Step1: remove wait() action"""
    tmp_instenvpairs = _remove_wait(tmp_instenvpairs)
    """Step2: remove move^grasp^keep sequence, there is no state change or any potential state change in this sequence"""
    tmp_instenvpairs = _remove_move_grasp_keep_seq(tmp_instenvpairs)
    """Step3: change the action order for moveto1 grasp1 moveto2 grasp2 XXX(1 2) and moveto1 grasp1 moveto2 XXX(1 2)"""
    tmp_instenvpairs = _change_action_order(tmp_instenvpairs)
    
    return tmp_instenvpairs               
           
def _remove_wait(instenvpairs):
    result_pairs = {}
    for k in sorted(instenvpairs.keys()):
        if (instenvpairs[k].getInst() is not None) and (instenvpairs[k].getInst().getInstVerb().lower() == 'wait'):
            continue
        else:
            prev_res_len = len(result_pairs)
            result_pairs[prev_res_len] = instenvpairs[k]
    return result_pairs

def _remove_move_grasp_keep_seq(instenvpairs):
    result_pairs = {}
    nodes_to_remove = []
    for k in sorted(instenvpairs.keys()):
        # the last element of this dict always have a None instruction, So here, if we consider sequence of length 3,
        # instead of checking k+2 reaching the end, we consider k+3 
        if k+3 in instenvpairs.keys():
            v1 = instenvpairs[k].getInst().getInstVerb().lower()
            v2 = instenvpairs[k+1].getInst().getInstVerb().lower()
            v3 = instenvpairs[k+2].getInst().getInstVerb().lower()
            if v1 == 'moveto' and v2 == 'grasp' and v3 == 'keep':
                inst3_args = instenvpairs[k+2].getInst().getInstArgs()
                test_predicate = StatePredicate()
                test_predicate.createFromDataStr(str(inst3_args[1])+' '+str(inst3_args[0])+' '+str(inst3_args[2]))
                if len( instenvpairs[k].getEnv().getNonExistencePreds([test_predicate]) ) == 0:
                    nodes_to_remove = nodes_to_remove+[k, k+1, k+2]
            else:
                continue
        else:
            break               
           
    for k in sorted(instenvpairs.keys()):
        if k in nodes_to_remove:
            continue
        else:
            prev_res_len = len(result_pairs)
            result_pairs[prev_res_len] = instenvpairs[k]
    return result_pairs
           
def _change_action_order(instenvpairs):           
    """step1: check pattern moveto1 grasp1 moveto2 grasp2 XXX(1 2)"""
    pattern_index_list = []
    for k in sorted(instenvpairs.keys()):
        # the last element of this dict always have a None instruction, So here, if we consider sequence of length 5,
        # instead of checking k+4 reaching the end, we consider k+5 
        if k+5 in instenvpairs.keys():
            v1 = instenvpairs[k].getInst().getInstVerb().lower()
            v1_arg1 = instenvpairs[k].getInst().getInstArgs()[0].lower()
            v2 = instenvpairs[k+1].getInst().getInstVerb().lower()
            v2_arg1 = instenvpairs[k+1].getInst().getInstArgs()[0].lower()
            v3 = instenvpairs[k+2].getInst().getInstVerb().lower()
            v3_arg1 = instenvpairs[k+2].getInst().getInstArgs()[0].lower()
            v4 = instenvpairs[k+3].getInst().getInstVerb().lower()
            v4_arg1 = instenvpairs[k+3].getInst().getInstArgs()[0].lower()
            v5 = instenvpairs[k+4].getInst().getInstVerb().lower()
            if v1=='moveto' and v2=='grasp' and v3=='moveto' and v4=='grasp' \
                    and v1_arg1==v2_arg1 and v3_arg1==v4_arg1 and v1_arg1 != v3_arg1:
                v5_args = instenvpairs[k+4].getInst().getInstArgs()
                if len(v5_args) == 2 and v5_args[0].lower() == v3_arg1 and v5_args[1].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3, k+4])
                elif len(v5_args) == 3 and v5_args[0].lower() == v3_arg1 and v5_args[2].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3, k+4])
                else:
                    pass     
            else:
                continue
        else:
            break           
           
    """step2: check pattern moveto1 grasp1 moveto2 XXX(1 2)"""
    for k in sorted(instenvpairs.keys()):
        if k+4 in instenvpairs.keys():
            v1 = instenvpairs[k].getInst().getInstVerb().lower()
            v1_arg1 = instenvpairs[k].getInst().getInstArgs()[0].lower()
            v2 = instenvpairs[k+1].getInst().getInstVerb().lower()
            v2_arg1 = instenvpairs[k+1].getInst().getInstArgs()[0].lower()
            v3 = instenvpairs[k+2].getInst().getInstVerb().lower()
            v3_arg1 = instenvpairs[k+2].getInst().getInstArgs()[0].lower()
            v4 = instenvpairs[k+3].getInst().getInstVerb().lower()
            if v1=='moveto' and v2=='moveto' and v3=='grasp' and \
                    v2_arg1==v3_arg1 and v1_arg1 != v2_arg1:
                v4_args = instenvpairs[k+3].getInst().getInstArgs()
                if len(v4_args) == 2 and v4_args[0].lower() == v2_arg1 and v4_args[1].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3])
                elif len(v4_args) == 3 and v4_args[0].lower() == v2_arg1 and v4_args[2].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3])
                else:
                    pass     
            else:
                continue
        else:
            break           
           
    result_pairs = {}
    for pattern in pattern_index_list:
        if len(pattern) == 4:
            result_pairs[pattern[0]] = instenvpairs[pattern[1]]
            result_pairs[pattern[1]] = instenvpairs[pattern[2]]
            result_pairs[pattern[2]] = instenvpairs[pattern[0]]
            result_pairs[pattern[3]] = instenvpairs[pattern[3]]
        elif len(pattern) == 5:
            result_pairs[pattern[0]] = instenvpairs[pattern[2]]
            result_pairs[pattern[1]] = instenvpairs[pattern[3]]
            result_pairs[pattern[2]] = instenvpairs[pattern[0]]
            result_pairs[pattern[3]] = instenvpairs[pattern[1]]
            result_pairs[pattern[4]] = instenvpairs[pattern[4]]
 
        else:
            raise Exception('sweep pattern length incorrect')           
           
    for k in instenvpairs.keys():
        if k in result_pairs.keys():
            continue
        else:
            result_pairs[k] = instenvpairs[k]#copy.deepcopy(instenvpairs[k])
    return result_pairs           
    
    
class Clause(object):
    def __init__(self):
        self.fileid = ''
        self.sent = ''
        self.instenvseq = {}
        
        self.dscr = ''
        
        self.instenvseq_for_evl = {}# this is for evaluation purpose. Cleaned the original instruction sequence to
                                       # remove noisy actions or change the order of noisy action sequences.

        self.verb = ''
        self.args = {}
        self.args_relations = []
        self.args_groundings = {}
        self.refined_args_groundings = {}
        
        self.verb_frame = ''

        self.hyponet_heu1 = None
        self.hyponet_heu2 = None
        self.hyponet_heu3 = None
        self.hyponet_heu4 = None
        
    def fillInstenvseqFromStartEnd(self, start_inst_id, end_inst_id, instenvseq):
        for i in range(start_inst_id, end_inst_id+1+1):
            self.instenvseq[i] = copy.deepcopy(instenvseq[i])
        self.instenvseq[end_inst_id+1].storeInst(None)
        
        self.instenvseq_for_evl = _clean_instenvpairs_to_get_seq_for_evl(self.instenvseq)
        
#         self.instenvseq_for_evl = self.instenvseq
        # further refine the instenvseq
        # given init env and final env, different planner may generate different
        # action sequence, but the final env could be exactly the same.
        # Here we generate the action sequence with our machine. And if the 
        # final env is exactly the same as the original, the generated action sequence
        # will replace the original one. Otherwise, nothing will happen.
        start_env = self.getStartEnv()
        end_env = self.getEndEnv()
        changed_terms = start_env.getNonExistencePreds(end_env.getAllPreds())
        for pred in end_env.getNonExistencePreds(start_env.getAllPreds()):
            pred_ = copy.deepcopy(pred)
            pred_.changeLabelToFalse()
            changed_terms.append(pred_)
        from symbolicplanner import SymbPlanner
        planner = SymbPlanner()
        sml = Simulator()
        sml.formSimulator()
        inst_seq = planner.execute_with_initenv_and_goal(start_env, changed_terms)
        instenvpairs = {}
        if len(inst_seq) > 0 and len(self.instenvseq_for_evl) == len(inst_seq)+1:
            for k in sorted(inst_seq.keys()):
                instenvpairs[len(instenvpairs)] = \
                    instenvPair(inst_seq[k], copy.deepcopy(start_env))
                present_env = sml.execute(inst_seq[k], start_env)
                start_env = present_env
            instenvpairs[len(instenvpairs)] = \
                instenvPair(None, copy.deepcopy(start_env))
         
            self.instenvseq_for_evl = instenvpairs

        
    def fillSentFromString(self, sent):
        self.sent = sent.lower()
        
    def fillClsDscrFromString(self, dscrstr):
        self.dscr = dscrstr
        self.verb = dscrstr.split('[')[0]
        tmp_args_rel_str = dscrstr.split('[')[1][:-1]
        args = tmp_args_rel_str.split('|')[:-1]
        tmp_rel_str = tmp_args_rel_str.split('|')[-1]

        for i in range(len(args)):
            self.args[i] = args[i]

        rel_s_locs = [ pos for pos, char in enumerate(tmp_rel_str) if char == '{' ]
        rel_e_locs = [ pos for pos, char in enumerate(tmp_rel_str) if char == '}' ]
        if len(rel_s_locs) != len(rel_e_locs):
            raise Exception('Clause:fillClsDscrFromString, number of { should be the same as }')
        else:
            for i in range(len(rel_s_locs)):
                rel_str = tmp_rel_str[rel_s_locs[i]+1 : rel_e_locs[i]]
                rel = ArgRelation()
                rel.createRelFromStr(rel_str)
                self.args_relations.append(rel)        

    def fillMapFromString(self, mapping_str):
        map_s_locs = [ pos for pos, char in enumerate(mapping_str) if char == '(' ]
        map_e_locs = [ pos for pos, char in enumerate(mapping_str) if char == ')' ]
        if len(map_s_locs) != len(map_e_locs):
            raise Exception('Clause:fillMapFromString, number of ( should be the same as )')
        else:
            for i in range(len(map_s_locs)):                
                tmp_map_str = mapping_str[map_s_locs[i]+1 : map_e_locs[i]]
                if tmp_map_str.split(',')[0] in self.args_groundings.keys():
                    self.args_groundings[tmp_map_str.split(',')[0]].append(tmp_map_str.split(',')[1])
                else:
                    self.args_groundings[tmp_map_str.split(',')[0]] = [tmp_map_str.split(',')[1]]
        if len(self.args_groundings)>0:
            self.extractRefinedArgsGroundings()

    def getArgGroundings(self):
        # tmp_grounding = {}
        # cls_args = self.getArgs()
        # for k in sorted(cls_args.keys()):
        #     if cls_args[k] in self.refined_args_groundings.keys():
        #         tmp_grounding['var'+str(len(tmp_grounding))] = \
        #                             self.getObjGround(cls_args[k])
        # return tmp_grounding
        return self.args_groundings
    
    def getObjGround(self, obj_name):
        if obj_name in self.refined_args_groundings.keys():
            return self.refined_args_groundings[obj_name][0].lower()
        else:
            return None    

    def getObjs_argonly(self):
        groundedObjs = []
        for i in sorted(self.refined_args_groundings.keys()):
            groundedObjs.append(self.getObjGround(i))
        return groundedObjs
    def getObjs_operated(self):
        groundedObjs = []
        insts = [instenv.getInst() for instenv in self.instenvseq.values() if instenv.getInst() is not None]
        for inst in insts:
            groundedObjs = groundedObjs + inst.getInstArgObjs()
        return list(set(groundedObjs))    
 
    def getObjs_hasdirectrelation(self):
        groundedObjs = []
        
        arg_objs = list(set([self.getObjGround(i) for i in sorted(self.refined_args_groundings.keys())]))#   list(set(self.args_groundings.values()))
        finalEnv = self.getEndEnv()
        for pred in finalEnv.getAllPreds():
            for predobj in pred.getPhyObjs():
                if predobj in arg_objs:
                    if pred.pred == 'state':
                        groundedObjs.append(pred.args[0])
                    else:
                        groundedObjs = groundedObjs + list(pred.getPhyObjs())
        return sorted(list(set(groundedObjs)))    

    def getObjs_allchanged(self):
        start_env = self.getStartEnv()
        end_env = self.getEndEnv()
        changed_preds = start_env.getNonExistencePreds(end_env.getAllPreds())
        changed_preds = changed_preds+end_env.getNonExistencePreds(start_env.getAllPreds())
        
        objs = []
        for pred in changed_preds:
            objs = objs+pred.getPhyObjs()
        return sorted(list(set(objs)))

    def getVerb(self):
        return str(self.verb).lower()
    
    def getStartEnv(self):
        if len(self.instenvseq) == 0:
            # return None means this clause doesn't include any action or env
            return None
        return self.instenvseq[min(self.instenvseq)].getEnv()

    def getEndEnv(self):
        if len(self.instenvseq) == 0:
            # return None means this clause doesn't include any action or env
            return None
        return self.instenvseq[max(self.instenvseq)].getEnv()
    
    def getGroundTruthInstSeq(self):
        instseq = []
        for k in sorted(self.instenvseq.keys()):
            if self.instenvseq[k].getInst() is None:
                continue
            else:
                instseq.append(self.instenvseq[k].getInst())
        return instseq     

    def getCleanedGroundTruthInstSeq(self):
        instseq = []
        for k in sorted(self.instenvseq_for_evl.keys()):
            if self.instenvseq_for_evl[k].getInst() is None:
                continue
            else:
                instseq.append(self.instenvseq_for_evl[k].getInst())
        return instseq
    
    def changeInstenvseq(self, newInstenvseq):
        self.instenvseq = newInstenvseq

    def getArgs(self):
        return self.args

    def extractVerbFrame(self):
        from kb import get_verb_frame
        self.verb_frame = get_verb_frame(self)

    def getRefinedArgsGroundings(self):
        return self.refined_args_groundings

    def extractRefinedArgsGroundings(self):
        everchanged_obj_set = self.getObjsWithStateEverChanged()
        self.refined_args_groundings = self._groundsInObjlist(self.args_groundings, everchanged_obj_set)      
    def getObjsWithStateEverChanged(self):
        everChangedObjs = []
        for k in sorted(self.instenvseq.keys()):
            if self.instenvseq[k].getInst() is not None:
                preEnv = self.instenvseq[k].getEnv()
                postEnv = self.instenvseq[k+1].getEnv()
                pos_changed_pred = preEnv.getNonExistencePreds(postEnv.getAllPreds())
                neg_changed_pred = postEnv.getNonExistencePreds(preEnv.getAllPreds())
                for pred in pos_changed_pred+neg_changed_pred:
                    everChangedObjs = everChangedObjs+pred.getPhyObjs()
        return [str(o) for o in set(everChangedObjs) if o != 'robot']       
    def _groundsInObjlist(self, args_groundings, obj_list):
        refined_args_groundings = {}
        for np in args_groundings.keys():
            np_orig_g = args_groundings[np]
            np_refined_g = []
            for g in np_orig_g:
                if g in obj_list:
                    np_refined_g.append(g)
            if len(np_refined_g)>0:
                refined_args_groundings[np] = np_refined_g
        return refined_args_groundings     
    def __str__(self):
        dscr = "file id:         "+self.fileid+'\n'+\
               "sentence:        "+self.sent+'\n'+\
               "verb:            "+self.verb+'\n'+\
               "grounds:         "+' '.join([str(self.args_groundings[k]) for k in sorted(self.args_groundings.keys())])+'\n'+\
               "instruction seq: "+' '.join([str(inst) for inst in self.getGroundTruthInstSeq()])+'\n'+\
               "cleaned inst seq:"+' '.join([str(inst) for inst in self.getCleanedGroundTruthInstSeq()])
               #"initial env:     "+' '.join([str(k) for k in self.getStartEnv().predlist])+'\n'+\
               #"end env:         "+' '.join([str(k) for k in self.getEndEnv().predlist])+'\n'+\
        return dscr               
                
class ArgRelation(object):
    def __init__(self):
        self.tail = ''
        self.head = ''
        self.relation = ''

    def createRelFromStr(self, rel_str):
        self.tail = rel_str.split('x')[0]
        self.relation = rel_str.split('->')[1]
        self.head = rel_str.split('x')[1].split('->')[0]                              
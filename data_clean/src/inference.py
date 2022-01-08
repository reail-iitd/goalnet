import copy
import random
import logging

from inducer import HypoNet
import constant
from symbolicplanner import SymbPlanner
from evl import Evaluator
from clausedata import instenvPair
from simulator import Simulator



logger = logging.getLogger(__name__)
"""Performing action sequence inference for testing data"""

class Inferencer(object):
    def __init__(self):
        self.test_data = None
        self.retrieved_hyponet = None
        self.instance_arg_grounding = None
        
    def performInfOnPara(self, test_data, kb, sml, opm):
        clslist = test_data.returnClauses()
        predict_result = {}
        starting_env = test_data.getBeginningEnv()
        for cls in clslist:
            cls_ = copy.deepcopy(cls)
            tmp_instenvseq = {0:instenvPair(None, copy.deepcopy(starting_env))}
            cls_.changeInstenvseq(tmp_instenvseq)
            
            actseq_seg = self.performInfOnCls(cls_, kb, sml, opm)
            
            for k in sorted(actseq_seg.keys()):
                if actseq_seg[k].getInst() is None:
                    continue
                nextEnv = sml.execute(actseq_seg[k].getInst(), starting_env)
                predict_result[len(predict_result)] = instenvPair(copy.deepcopy(actseq_seg[k].getInst()), copy.deepcopy(starting_env)) 
                starting_env = nextEnv
        predict_result[len(predict_result)] = instenvPair(None, copy.deepcopy(starting_env))
        return predict_result
    
    def performInfOnCls(self, test_data, kb, sml, opm=None):
        predict_result = {}
        if kb.checkVerbStrcExists(test_data):
            logger.debug('clause info: '+test_data.fileid)#str(test_data))
            self.test_data = test_data
            self._formArgGrounding()
            #self.retrieved_hyponet = copy.deepcopy(kb.getHypoNetForCls(test_data))
            self.retrieved_hyponet = kb.getHypoNetForCls(test_data)
            
            if self.retrieved_hyponet.getSize() > 8000:
                logger.info('size of the hypo space for this action structure is larger than 8000, abandoned')
                return predict_result
            if len(self.retrieved_hyponet.botNodes.keys()) == 0:
                logger.info('knowledge exist but no nodes satisfy arguments affordances')
                return {}
#             if test_data.fileid == '171.clauses' and test_data.verb_frame == 'put+var0':
#                 print('test')
#             else:
#                 return predict_result
            
            if constant.test_actseq_gen_crit == 'upper_bound':
                final_seq = self._gen_action_seq_upper_bound()
            elif constant.test_actseq_gen_crit == 'memory_based':
                final_seq = self._gen_action_seq_memory_based()
            elif constant.test_actseq_gen_crit == 'most_general_node':
                final_seq = self._gen_action_seq_most_general()
            elif constant.test_actseq_gen_crit == 'highest_node_freq':
                final_seq = self._gen_action_seq_highest_node_freq()
            elif constant.test_actseq_gen_crit == 'with_optimizer':
                final_seq = self._gen_action_seq_with_optimizer(opm)
            elif constant.test_actseq_gen_crit == 'with_optimizer_classificationbased':
                final_seq = self._gen_action_seq_with_optimizer_classificationbased(opm)
            if len(final_seq) == 0:
                return {}
            starting_env = test_data.getStartEnv()
            for k in sorted(final_seq.keys()):
                nextEnv = sml.execute(final_seq[k], starting_env)
                predict_result[len(predict_result)] = instenvPair(final_seq[k],copy.deepcopy(starting_env))
                starting_env = nextEnv
            predict_result[len(predict_result)] = instenvPair(None, copy.deepcopy(starting_env))
                            
            return predict_result
        else:
            logger.debug('knowledge base does not have the verb structure')
            logger.debug(str(test_data))
            return predict_result
        
    
    def _formArgGrounding(self):
        """self.test_data must be an object of clause class"""
        self.instance_arg_grounding = self.test_data.getArgGroundings()
 
    def _modify_instance_preds_with_new_arg(self, instance):
        new_preds = copy.deepcopy(instance.instance_preds)
        new_arg_map = {}
        
        for var_key in self.instance_arg_grounding.keys():
            new_ground = self.instance_arg_grounding[var_key]
            old_ground = instance.instance_args[var_key]
            new_arg_map[old_ground] = new_ground 
            
        for new_pred in new_preds:
            new_pred.changeobj(new_arg_map)
        return new_preds
        
    def _hypo_obj_rand_instantiate(self, hypo):
        new_preds = copy.deepcopy(hypo.vard_preds)
        old_objs=[]
        for pred in new_preds:
            old_objs = old_objs+pred.getPhyObjs()
        old_objs = list(set(old_objs))
        new_arg_map = {}
        env_obj_list = copy.deepcopy(self.test_data.instenvseq[min(self.test_data.instenvseq.keys())].getEnv().getObj())
        for old_obj in old_objs:
            if old_obj.startswith('var'):
                new_arg_map[old_obj] = self.instance_arg_grounding[old_obj]
                for obj in env_obj_list:
                    if obj.idfer == self.instance_arg_grounding[old_obj]:
                        env_obj_list.remove(obj)
                        break
        for old_obj in old_objs:
            if old_obj.startswith('obj'):
                old_affords = hypo.var_affords[old_obj]
                not_exist_sameafford_obj = True
                for obj in env_obj_list:
                    if set(obj.getObjAfford()) == set(old_affords):
                        new_arg_map[old_obj] = obj.idfer
                        env_obj_list.remove(obj)
                        not_exist_sameafford_obj = False
                        break
                if not_exist_sameafford_obj:
                    logger.debug('some obj variable in preds cannot be grounded with same affords')
                    return []
        for new_pred in new_preds:
            new_pred.changeobj(new_arg_map)
        return new_preds
                        

    def _genCandiGoals(self, hypo):
        goal_candis = {}
        if len(hypo.instances) == 0:
            return goal_candis
        
        env_obj_string = self.test_data.instenvseq[min(self.test_data.instenvseq.keys())].getEnv().getObjsStr()
        for instance in hypo.instances:
            new_preds = self._modify_instance_preds_with_new_arg(instance)
            new_objs = []
            for pred in new_preds:
                new_objs = new_objs+pred.getPhyObjs()
            new_objs = list(set(new_objs))
            all_obj_in = True
            for new_obj in new_objs:
                if env_obj_string.find(new_obj) == -1:
                    all_obj_in = False
                    break
                else:
                    continue
            if all_obj_in:
                tmp_goal = candiGoal(new_preds, str(hypo), instance.instance_var_mapping)
                if str(tmp_goal) in goal_candis.keys():
                    goal_candis[str(tmp_goal)].increaseFreqBy1()
                    goal_candis[str(tmp_goal)].train_instances.append(instance)
                else:
                    goal_candis[str(tmp_goal)] = tmp_goal
                    goal_candis[str(tmp_goal)].train_instances.append(instance)
#         new_preds = self._hypo_obj_rand_instantiate(hypo)
#         if len(new_preds) > 0:
#             tmp_goal = candiGoal(new_preds, str(hypo))
#             if str(tmp_goal) in goal_candis.keys():
#                 goal_candis[str(tmp_goal)].increaseFreqBy1()
#             else:
#                 goal_candis[str(tmp_goal)] = tmp_goal
        return goal_candis

    def _gen_action_seq_memory_based(self):
#         random_choose_candidates = False
#         calculate_upper_bound = True
#          
#         if calculate_upper_bound:
#             gt_inst_seq = dict(enumerate(self.test_data.getCleanedGroundTruthInstSeq()))
#             evaluator = Evaluator()
#             score = 0
        
        planner = SymbPlanner()        
        final_seq = {}
        goal_candis = {}
        
        for hypo_id in sorted(self.retrieved_hyponet.botNodes.keys()):
            botNode = self.retrieved_hyponet.botNodes[hypo_id]
            tmp_goal_candis = self._genCandiGoals(botNode)
            if len(tmp_goal_candis) > 0:
                for k in sorted(tmp_goal_candis.keys()):
                    if k in goal_candis.keys():
                        goal_candis[k].increaseFreqBy1()
                    else:
                        inst_seq = planner.execute_with_initenv_and_goal(\
                                    self.test_data.getStartEnv(), \
                                    tmp_goal_candis[k].instance_goal)
                        tmp_goal_candis[k].candi_act_seq = inst_seq
                        if len(inst_seq) > 0:
                            goal_candis[k] = tmp_goal_candis[k]
            else:
                continue
        if len(goal_candis) == 0:
            return final_seq
            
        max_freq = 0
        for k in sorted(goal_candis.keys()):
            if goal_candis[k].freq > max_freq:
                max_freq = goal_candis[k].freq
                final_seq = goal_candis[k].get_act_seq()
        return final_seq
#         if random_choose_candidates:
#             import random
#             final_seq = random.choice([goal_candi.get_act_seq() for goal_candi in goal_candis.values()])
#          
#         if calculate_upper_bound:
#             tmp_score = 0
#             for goal_candi in goal_candis.values():
#                 aseq = goal_candi.get_act_seq()
#                 tmp_score = 1 - evaluator.levenshtein(gt_inst_seq, aseq)\
#                                             / (max(len(gt_inst_seq), len(aseq))+constant.epsilon)
#                 if tmp_score > score:
#                     score = tmp_score
#                     final_seq = aseq
#          
#         return final_seq
 
    def _gen_action_seq_most_general(self):
        planner = SymbPlanner()
        
        for level_k in sorted(self.retrieved_hyponet.hypos.keys()):
            hypos = self.retrieved_hyponet.hypos[level_k]
            goal_candis = {}
            for hypo in hypos.values():
                tmp_goal_candis = self._genCandiGoals(hypo)
                for k in sorted(tmp_goal_candis.keys()):
                    if k in goal_candis.keys():
                        goal_candis[k].increaseFreqBy1()
                    else:
                        goal_candis[k] = tmp_goal_candis[k]
            if len(goal_candis) == 0:
                continue
            inst_seq_freq_dict = {}
            for goal_candi in goal_candis.values():
                inst_seq = planner.execute_with_initenv_and_goal(\
                                    self.test_data.getStartEnv(), goal_candi.instance_goal)
                if len(inst_seq) > 0:
                    inst_seq_str = ' '.join([str(inst_seq[k]) for k in sorted(inst_seq.keys())])
                    if inst_seq_str in inst_seq_freq_dict.keys():
                        inst_seq_freq_dict[inst_seq_str]['freq'] = \
                            inst_seq_freq_dict[inst_seq_str]['freq']+goal_candi.freq
                    else:
                        inst_seq_freq_dict[inst_seq_str] = {'freq':goal_candi.freq, 'instseq':inst_seq}    
            max_freq = 0
            most_freq_inst_seq = {}
            for k in sorted(inst_seq_freq_dict.keys()):
                if inst_seq_freq_dict[k]['freq'] > max_freq:
                    max_freq = inst_seq_freq_dict[k]['freq']
                    most_freq_inst_seq = inst_seq_freq_dict[k]['instseq']
            if len(most_freq_inst_seq) > 0:
                return most_freq_inst_seq 
        return{}                
    def _gen_action_seq_highest_node_freq(self):
        planner = SymbPlanner()
        hypos = self.retrieved_hyponet.returnAllHypos()
        
        hypo_freq_dict = {}
        for hypo in hypos:
            freq = hypo.freq
            if freq in hypo_freq_dict:
                hypo_freq_dict[freq].append(hypo)
            else:
                hypo_freq_dict[freq] = [hypo]
        
        for freq in reversed(sorted(hypo_freq_dict.keys())):
            hypos = hypo_freq_dict[freq]
            goal_candis = {}
            for hypo in hypos:
                tmp_goal_candis = self._genCandiGoals(hypo)
                for k in sorted(tmp_goal_candis.keys()):
                    if k in goal_candis.keys():
                        goal_candis[k].increaseFreqBy1()
                    else:
                        goal_candis[k] = tmp_goal_candis[k]
            if len(goal_candis) == 0:
                continue
            inst_seq_freq_dict = {}
            for goal_candi in goal_candis.values():
                inst_seq = planner.execute_with_initenv_and_goal(\
                                self.test_data.getStartEnv(), goal_candi.instance_goal)
                if len(inst_seq) > 0:
                    inst_seq_str = ' '.join([str(inst_seq[k]) for k in sorted(inst_seq.keys())])
                    if inst_seq_str in inst_seq_freq_dict.keys():
                        inst_seq_freq_dict[inst_seq_str]['freq'] = \
                            inst_seq_freq_dict[inst_seq_str]['freq']+goal_candi.freq
                    else:
                        inst_seq_freq_dict[inst_seq_str] = {'freq':goal_candi.freq, 'instseq':inst_seq}
            max_freq = 0
            most_freq_inst_seq = {}
            for k in sorted(inst_seq_freq_dict.keys()):
                if inst_seq_freq_dict[k]['freq'] > max_freq:
                    max_freq = inst_seq_freq_dict[k]['freq']
                    most_freq_inst_seq = inst_seq_freq_dict[k]['instseq']
            if len(most_freq_inst_seq) > 0:
                return most_freq_inst_seq
        
        return{}
        
    def _gen_action_seq_upper_bound(self):
        gt_inst_seq = self.test_data.instenvseq_for_evl#dict(enumerate(self.test_data.getCleanedGroundTruthInstSeq()))
        
        evaluator = Evaluator()
        planner = SymbPlanner()
        sml = Simulator()
        sml.formSimulator()
        score = 0
        final_seq = {}
        neg_hypo_ids = []
        
        logtext = self.test_data.verb+' <'+str(self.test_data.dscr)+'> ('+str(self.test_data.refined_args_groundings)+')'+'\"'+str(self.test_data.verb_frame)+'\"'
        
        
        debug_info = {}
        
        for level_k in sorted(self.retrieved_hyponet.hypos.keys()):
            hypos = self.retrieved_hyponet.hypos[level_k]
            for k in sorted(hypos.keys()):
                if k in neg_hypo_ids:
                    continue
                else:
                    hypo = hypos[k]
                    goal_candis = self._genCandiGoals(hypo)
                    if len(goal_candis) > 0:
                        neg_flag = True
                        for goal_candi in goal_candis.values():
                            inst_seq = planner.execute_with_initenv_and_goal(\
                                    self.test_data.getStartEnv(), goal_candi.instance_goal)
                            goal_candi.candi_act_seq = inst_seq
                            if len(inst_seq) > 0:
                                neg_flag = False
                                
                                instenvpairs = {}
                                start_env = self.test_data.getStartEnv()
                                for i in sorted(inst_seq.keys()):
                                    instenvpairs[len(instenvpairs)] = \
                                        instenvPair(inst_seq[i], copy.deepcopy(start_env))
                                    present_env = sml.execute(inst_seq[i], start_env)
                                    start_env = present_env
                                instenvpairs[len(instenvpairs)] = \
                                    instenvPair(None, copy.deepcopy(start_env))
                                
                                #tmp_score = evaluator.evl_distance(gt_inst_seq, instenvpairs)
                                tmp_score = evaluator.evl_jacindex(gt_inst_seq, instenvpairs)
                                if tmp_score in debug_info.keys():
                                    debug_info[tmp_score].append({'hypo':hypo,'goal_candi':goal_candi})
                                else:
                                    debug_info[tmp_score] = [{'hypo':hypo,'goal_candi':goal_candi}]
                                logtext = logtext+'\n'+"{0:.5f}".format(tmp_score)+' (freq:'+str(goal_candi.freq)+') '+str(goal_candi)+'======'
                                if tmp_score > score:
                                    score = tmp_score
                                    final_seq = inst_seq
                                    
#                                     if score == 1:
#                                         return final_seq
                        if neg_flag:
                            neg_hypo_ids.append(str(k))
                    else:
                        neg_hypo_ids.append(str(k))
                        continue
            tmp_neg_hypos = self.retrieved_hyponet.get_parents_by_ids(neg_hypo_ids)
            neg_hypo_ids = copy.deepcopy(tmp_neg_hypos)
        logger.debug(logtext+'\n')
        return final_seq       

    def _gen_action_seq_with_optimizer(self, opm):
        final_scores = {}
        score_goalcandi_dict = {}
        no_aseq_hypo_list = []
        all_hypos = self.retrieved_hyponet.returnAllHypos()
        """get the maximum frequency of goal candidates"""
        # calculate the global feature max_candi_freq used by the optimizer
        max_candi_freq = 0
        for hypo in all_hypos:
            candis = self._genCandiGoals(hypo)
            for c in candis.values():
                if c.freq> max_candi_freq:
                    max_candi_freq = c.freq
                    
        for hypo in all_hypos:
            goalcandis = self._genCandiGoals(hypo)
            if len(goalcandis) == 0:
                continue
            for k in sorted(goalcandis.keys()):
                goalcandi = goalcandis[k]
#                 complex_hypo_preds = set([str(pred) for pred in goalcandi.instance_goal])
#                 # no_aseq_hypo_set records the hypotheses that generate empty action sequences,
#                 # where the planner cannot generate a sequence to achieve the goal.
#                 # if complex_hypo_preds is a larger set of any hypothesis in the no_aseq_hypo_set,
#                 # the complex_hyp_preds is also unachievable for the planner, just pass it.
#                 if check_parentship(no_aseq_hypo_list, complex_hypo_preds):
#                     continue
                # the opm.predict_score for different opm_type/s are implemented differently, pay attention.
                tmp_score, tmp_seq = opm.predict_score(max_candi_freq, goalcandi, self.test_data,\
                                                       hypo, self.retrieved_hyponet)
#                 if len(tmp_seq) == 0:
#                     no_aseq_hypo = set([str(pred) for pred in goalcandi.instance_goal])
#                     no_aseq_hypo_list.append(no_aseq_hypo)
                goalcandi.candi_act_seq = tmp_seq
                if tmp_score in score_goalcandi_dict.keys():
                    score_goalcandi_dict[tmp_score].append(goalcandi)
                else:
                    score_goalcandi_dict[tmp_score] = [goalcandi]
         
                if tmp_score in final_scores.keys():
                    final_scores[tmp_score].append(tmp_seq)
                else:
                    final_scores[tmp_score] = [tmp_seq]
         
        seq_score_dict = {}            
        for k in final_scores.keys():
            for seq in final_scores[k]:
                seq_str = ' '.join([str(seq[k]) for k in sorted(seq.keys())])
                if seq_str in seq_score_dict.keys():
                    seq_score_dict[seq_str]['score'] = seq_score_dict[seq_str]['score']+k
                else:
                    seq_score_dict[seq_str] = {'score':k, 'seq':seq}
        final_inst_seq = {}
        max_score = -1000
        for k in seq_score_dict.keys():
            if seq_score_dict[k]['score']>max_score and len(seq_score_dict[k]['seq'])>0:
                max_score = seq_score_dict[k]['score']
                final_inst_seq = seq_score_dict[k]['seq']
         
        if self.test_data.verb_frame=='put+var0+var1':
            print('test')
        return final_inst_seq

#         for k in reversed(sorted(final_scores.keys())):
#             if len(final_scores[k]) > 0:
#                 for aseq in final_scores[k]:
#                     if len(aseq)>0:
#                         return aseq
        return {}
    
    def _gen_action_seq_with_optimizer_classificationbased(self, opm):
        from classifier_optimizer import Classifier_optimizer
        if isinstance(opm, Classifier_optimizer):
            return self._gen_action_seq_with_optimizer(opm)
        else:
            raise Exception('while using classification based optimizer, the opm_type must be within sgd_lsvm/sgd_logreg/sgd_perceptron') 
        
        
        
def check_parentship(no_aseq_hypo_list, complex_hypo):
    for no_aseq_hypo in no_aseq_hypo_list:
        if no_aseq_hypo.issubset(complex_hypo):
            return True
        else:
            continue
    return False
        
    
class candiGoal(object):
    def __init__(self, instance_goal = [], node_id = '', instance_var_map = {}):
        self.instance_goal = instance_goal
        self.crpd_node_id = node_id
        self.candi_act_seq = {}
        self.instance_var_map = instance_var_map
        self.freq = 1
        self.train_instances = []
    def increaseFreqBy1(self):
        self.freq = self.freq+1
    def get_preds(self):
        return self.instance_goal
    def get_act_seq(self):
        return self.candi_act_seq
    def __str__(self):
        return ''.join(sorted([str(pred) for pred in self.instance_goal]))
import constant
import subprocess
import copy
from instruction import Instruction

def action_seq_clean(action_seq):
    """step1: check pattern moveto1 grasp1 moveto2 grasp2 XXX(1 2)"""
    pattern_index_list = []
    for k in sorted(action_seq.keys()):
        if k+4 in action_seq.keys():
            v1 = action_seq[k].getInstVerb().lower()
            v1_arg1 = action_seq[k].getInstArgs()[0].lower()
            v2 = action_seq[k+1].getInstVerb().lower()
            v2_arg1 = action_seq[k+1].getInstArgs()[0].lower()
            v3 = action_seq[k+2].getInstVerb().lower()
            v3_arg1 = action_seq[k+2].getInstArgs()[0].lower()
            v4 = action_seq[k+3].getInstVerb().lower()
            v4_arg1 = action_seq[k+3].getInstArgs()[0].lower()
            v5 = action_seq[k+4].getInstVerb().lower()
            if v1=='moveto' and v2=='grasp' and v3=='moveto' and v4=='grasp' \
                    and v1_arg1==v2_arg1 and v3_arg1==v4_arg1 and v1_arg1 != v3_arg1:
                v5_args = action_seq[k+4].getInstArgs()
                if len(v5_args) == 2 and v5_args[0].lower() == v3_arg1 \
                        and v5_args[1].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3, k+4])
                elif len(v5_args) == 3 and v5_args[0].lower() == v3_arg1 \
                        and v5_args[2].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3, k+4])
                else:
                    pass     
            else:
                continue
        else:
            break
        
    """step2: check pattern moveto1 moveto2 grasp2  XXX(2 1)"""
    for k in sorted(action_seq.keys()):
        if k+3 in action_seq.keys():
            v1 = action_seq[k].getInstVerb().lower()
            v1_arg1 = action_seq[k].getInstArgs()[0].lower()
            v2 = action_seq[k+1].getInstVerb().lower()
            v2_arg1 = action_seq[k+1].getInstArgs()[0].lower()
            v3 = action_seq[k+2].getInstVerb().lower()
            v3_arg1 = action_seq[k+2].getInstArgs()[0].lower()
            v4 = action_seq[k+3].getInstVerb().lower()
            if v1=='moveto' and v2=='moveto' and v3=='grasp' \
                    and v2_arg1==v3_arg1 and v1_arg1 != v2_arg1:
                v4_args = action_seq[k+3].getInstArgs()
                if len(v4_args) == 2 and v4_args[0].lower() == v2_arg1 \
                        and v4_args[1].lower() == v1_arg1:
                    pattern_index_list.append([k, k+1, k+2, k+3])
                elif len(v4_args) == 3 and v4_args[0].lower() == v2_arg1 \
                        and v4_args[2].lower() == v1_arg1:
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
            result_pairs[pattern[0]] = action_seq[pattern[1]]
            result_pairs[pattern[1]] = action_seq[pattern[2]]
            result_pairs[pattern[2]] = action_seq[pattern[0]]
            result_pairs[pattern[3]] = action_seq[pattern[3]]
        elif len(pattern) == 5:
            result_pairs[pattern[0]] = action_seq[pattern[2]]
            result_pairs[pattern[1]] = action_seq[pattern[3]]
            result_pairs[pattern[2]] = action_seq[pattern[0]]
            result_pairs[pattern[3]] = action_seq[pattern[1]]
            result_pairs[pattern[4]] = action_seq[pattern[4]]

        else:
            raise Exception('sweep pattern length incorrect')
    for k in action_seq.keys():
        if k in result_pairs.keys():
            continue
        else:
            result_pairs[k] = action_seq[k]
    return result_pairs

def quick_filter(goal_state):
    refined_goal_state = []
    
    for pred in goal_state:
        """filter step1: certain state only belongs to specific object, example: state fridge rightdoorisopen. There's no state kettle rightdoorisopen"""
        if pred.getPred() == 'state':
            if filter_state_x_y(pred):
                refined_goal_state.append(pred)
                pass
            else:
                if pred.getTfLabel() is False:
                    pass
                else:
                    return False
        """filter step2: not belongs to illegal predicate set"""
        if filter_illegal_pred(pred):
            refined_goal_state.append(pred)
            pass
        else:
            if pred.getTfLabel() is False:
                pass
            else:
                return False
        refined_goal_state = list(set(refined_goal_state))
    return refined_goal_state


graspableset = ['boiledegg_1', 'ramen_1', 'salt_1', 'icecream_1', 'bagofchips_1', \
        'beer_1', 'plate_1', 'cd_2', 'xbox_1', 'plate_2', 'pillow_1', 'pillow_3',\
        'tv_1remote_1', 'xboxcontroller_1', 'book_2', 'pillow_4', 'cd_1', 'pillow_2',\
        'book_3', 'bowl_1', 'book_1', 'coke_1', 'instantramen_1', 'fork_1', 'spoon_1', 'garbagebag_1',\
        'garbagebin_1', 'kettle', 'mug_1', 'energydrink_1', 'glass_1', 'canadadry_1', 'longcup_1', 'longcup_2',\
        'syrup_1', 'syrup_2']   
    
def filter_state_x_y(predicate):
    """the predicate has to be state x y"""
    arg0 = predicate.getArgs()[0]
    arg1 = predicate.getArgs()[1]
    if arg1 in ['microwaveison','doorisopen'] and arg0 != 'microwave':
        return False
    if arg1 in ['waterdispenserisopen','leftdoorisopen','rightdoorisopen'] and arg0 != 'fridge':
        return False
    if arg1 in ['channel1','channel2','channel3','channel4','channel5','channel6','volume','ison'] and arg0 != 'tv_1':
        return False
    if arg1 == 'tapison' and arg0 != 'sinkknob':
        return False
    if arg1 in ['stovefire1','stovefire2','stovefire3','stovefire4'] and arg0 != 'stove':
        return False
    if arg1 in ['rightdoorisopen','leftdoorisopen'] and arg0 != 'fridge':
        return False
    if arg1 == 'isopen' and arg0 != 'bagofchips_1':
        return False
    if arg1 in ['fork','spoon','cd', 'water', 'temperaturehigh'] and (arg0 not in graspableset):
        return False
    if arg1 in ['water', 'coffee', 'coke', 'energydrink', 'icecream', 'chocolate', 'canadadry', 'vanilla', 'salt', 'ramen', 'egg'] and \
        arg0 not in ['boiledegg_1', 'ramen_1', 'icecream_1', 'plate_1', 'plate_2', 'bowl_1', 'instantramen_1', 'spoon_1', 'kettle', \
                     'mug_1', 'energydrink_1', 'glass_1', 'canadadry_1', 'longcup_1', 'longcup_2', 'syrup_1', 'syrup_2', 'fork_1']:
        return False
    return True
    
def filter_illegal_pred(pred):
    pred_str = pred.getPredStr()
    if pred.getPred() == 'grasping':
        if pred.getArgs()[0] != 'robot' or (pred.getArgs()[1] not in graspableset):
            return False
    if pred.getPred() == 'near':
        if pred.getArgs()[0] != 'robot':
            return False
    if pred.getPred() == 'in':
        if pred.getArgs()[0] not in graspableset or (pred.getArgs()[1] not in ['fridgeleft', 'fridgeright', 'xbox_1', 'instantramen_1', \
                                                                               'garbagebag_1', 'garbagebin_1', 'fridge', 'microwave', \
                                                                               'kettle', 'mug_1', 'glass_1', 'longcup_1', 'longcup_2', \
                                                                               'plate_1', 'plate_2', 'bowl_1']):
            return False
    if pred.getPred() == 'on':
        if pred.getArgs()[0] not in graspableset or (pred.getArgs()[1] not in ['garbagebin_1', 'shelf_1', 'shelf_2', 'snacktable_1', \
                                                                               'armchair_4', 'counter1_1', 'stovefire_4', \
                                                                               'stovefire_2', 'stovefire_3', 'sink', 'stovefire_1',\
                                                                               'armchair_3', 'counter_1', 'studytable_1', 'armchair_1',\
                                                                               'loveseat_2', 'loveseat_1', 'armchair_2', 'coffeetable_1', 'tvtable_1']):
            return False
    return True


class SymbPlanner(object):
    """description of class"""
    def __init__(self):
        self.domain_file = constant.domain_file
        self.tmp_problem_file = constant.tmp_problem_file
        self.planner = constant.planner
        
    def execute_with_initenv_and_goal(self,init_env, goal_state):
        if constant.use_cached_planning_result:
            prob_str = self._form_env_str(init_env)+self._form_goal_str(goal_state)
            if prob_str in constant.get_cached_planning_result().keys():
                return action_seq_clean(constant.get_cached_planning_result()[prob_str])            
            else:
                filter_result = quick_filter(goal_state)
                if filter_result is not False:
                    goal_state = filter_result
                else:
                    inst_seq = {}
                    constant.append_new_planning_result(prob_str, inst_seq)
                    return {}
                
                self._form_problem_file(init_env, goal_state)
                proc = subprocess.Popen([self.planner, self.domain_file, self.tmp_problem_file,'-T','500','-t','15','-P','0'],stdout=subprocess.PIPE,)
                planresult = proc.communicate()[0].decode("utf-8")
                plansteps = [i for i in planresult.split('\n') if i.startswith('STEP')]
                inst_seq = {}
                for i in range(len(plansteps)):
                    inst = Instruction()
                    inst.createInstFromPlanResult(plansteps[i][plansteps[i].find(':')+1 : ])
                    inst_seq[i] = inst
                inst_seq = action_seq_clean(inst_seq)
                constant.append_new_planning_result(prob_str, inst_seq)
                return inst_seq
        else:
            filter_result = quick_filter(goal_state)
            prob_str = self._form_env_str(init_env)+self._form_goal_str(goal_state)
            if filter_result is not False:
                goal_state = filter_result
            else:
                inst_seq = {}
                #constant.append_new_planning_result(prob_str, inst_seq)
                return {}
            
            self._form_problem_file(init_env, goal_state)
            proc = subprocess.Popen([self.planner, self.domain_file, self.tmp_problem_file,'-T','500','-t','15','-P','0'],stdout=subprocess.PIPE,)
            planresult = proc.communicate()[0].decode("utf-8")
            plansteps = [i for i in planresult.split('\n') if i.startswith('STEP')]
            inst_seq = {}
            for i in range(len(plansteps)):
                inst = Instruction()
                inst.createInstFromPlanResult(plansteps[i][plansteps[i].find(':')+1 : ])
                inst_seq[i] = inst
            #inst_seq = action_seq_clean(inst_seq)
            constant.append_new_planning_result(prob_str, inst_seq)
            return inst_seq           
            
    def _form_problem_file(self, init_env, goal_state):#, neg_goal_state):
        prb_str = ""
        obj_str = ""
        initenv_str = ""
        goal_str = ""

        prbfile_id = open(self.tmp_problem_file, "w")
        prbfile_id.write("(define \n")
        
        prb_str = self._form_prb_str()
        prbfile_id.write(prb_str)

        obj_str = self._form_obj_str(init_env)
        prbfile_id.write(obj_str)

        initenv_str = self._form_env_str(init_env)
        prbfile_id.write(initenv_str)

        goal_str = self._form_goal_str(goal_state)#, neg_goal_state)
        prbfile_id.write(goal_str)

        prbfile_id.write(")\n")
        prbfile_id.close()

    def _form_prb_str(self):
        return "(problem tmp)\n"

    def _form_obj_str(self, env):
        return "(:objects"+env.getObjsStr()+" on in near)\n"            
            
    def _form_env_str(self, env):
        return "(:init "+env.getPredsStr()+")\n"        
    
    def _form_goal_str(self, goalpreds):#, neg_goalpreds):
        return "(:goal (AND "+" ".join(sorted([pred.getPredStr() for pred in goalpreds]))+"))\n"
import copy
import itertools
import pickle
import logging

from kb import VerbSemKB
import constant
from symbolicplanner import SymbPlanner
from statepredicate import StatePredicate

logger = logging.getLogger(__name__)

def check_h_sibof_l(h_id, l_id):
    if len(h_id)+1 != len(l_id):
        return False
    temp = l_id[:]
    try:
        for v in h_id:
            temp.remove(v)
        return True
    except:
        return False

class VerbSemInducer(object):
    def __init__(self):
        self.data = None
        self.kb = None
        
    def induceFromFile(self, allData, trainId, dump_file_name = None):
        if constant.load_existing_kb:
            logger.info('loading knowledge base: '+constant.precomputed_kb)
            self.kb = pickle.load(open(constant.precomputed_dataset_dir+constant.precomputed_kb,"rb"))
            logger.info('knowledge base loaded!')
        else:
            self.kb = VerbSemKB()
            self.data = [allData[i] for i in trainId]            
            test_print_id = 0
            file_id = 0
            for dt in self.data:
                cls_list = dt.returnClauses()
                file_id = file_id+1
                if file_id%100 == 0:
                    print('\n')
                print('induce on '+str(file_id)+'th file')
                for cls in cls_list:
                    test_print_id = test_print_id+1
                    logger.info("inducing hypo net from cls "+str(cls.fileid)+' for '+cls.verb)
                    logger.info('test id: '+str(test_print_id))
                    if constant.node_chosen_heu == 'Heuristic1':
                        hn = copy.deepcopy(cls.hyponet_heu1)
                    elif constant.node_chosen_heu == 'Heuristic2':
                        hn = copy.deepcopy(cls.hyponet_heu2)
                    elif constant.node_chosen_heu == 'Heuristic3':
                        hn = copy.deepcopy(cls.hyponet_heu3)
                    elif constant.node_chosen_heu == 'Heuristic4':
                        hn = copy.deepcopy(cls.hyponet_heu4)
                    if len(hn.botNodes) == 0:
                        continue
                    else:
                        if self.kb.checkVerbExists(cls):
                            if self.kb.checkVerbStrcExists(cls):
                                self.kb.mergeNet(cls, hn)
                            else:
                                self.kb.createStrc(cls, hn)
                        else:
                            self.kb.createTerm(cls, hn)
            if dump_file_name is None:
                pass
            else:
                pickle.dump(self.kb, open(dump_file_name, 'wb'))
                logger.info('induce finish!')

    def induceFromClause(self, allData, trainId, dump_file_name = None):
        if constant.load_existing_kb:
            self.kb = pickle.load(open(constant.precomputed_dataset_dir+constant.precomputed_kb,"rb"))
            logger.info('knowledge base loaded!')

        else:
            self.kb = VerbSemKB()
            self.data = [allData[i] for i in trainId]
            test_print_id = 0
            for dt in self.data:
                test_print_id = test_print_id+1
                logger.info('inducing hypo net from cls '+str(dt.fileid)+' for '+dt.verb)
                logger.info(str(test_print_id)+'th induction data')
                print('induce on '+str(test_print_id)+'th cls')
                
                if constant.node_chosen_heu == 'Heuristic1':
                    hn = copy.deepcopy(dt.hyponet_heu1)
                elif constant.node_chosen_heu == 'Heuristic2':
                    hn = copy.deepcopy(dt.hyponet_heu2)
                elif constant.node_chosen_heu == 'Heuristic3':
                    hn = copy.deepcopy(dt.hyponet_heu3) 
                elif constant.node_chosen_heu == 'Heuristic4':
                    hn = copy.deepcopy(dt.hyponet_heu4)      
                    
                       
                if len(hn.botNodes) == 0:
                    continue
                else:
                    if constant.test_actseq_gen_crit in ['with_optimizer','upper_bound']:
                        hn.hypos = {min(hn.hypos.keys()): hn.hypos[min(hn.hypos.keys())]}   
                    if self.kb.checkVerbExists(dt):
                        if self.kb.checkVerbStrcExists(dt):
                            self.kb.mergeNet(dt, hn)
                        else:
                            self.kb.createStrc(dt, hn)
                    else:
                        self.kb.createTerm(dt, hn)
            if dump_file_name is None:
                pass
            else:
                pickle.dump(self.kb, open(dump_file_name, 'wb'))
                logger.info('induce finish!')                        
                        
                                            
class HypoNet(object):
    def __init__(self):
        self.hypos = {}
        self.links = {}
        self.botNodes = {}
        self.clauses = {}

    def return_highest_freq(self):
        highest_freq = 0
        for lk in self.hypos.keys():
            nodes = self.hypos[lk]
            for node in nodes.values():
                if node.freq > highest_freq:
                    highest_freq = node.freq
        return highest_freq
        
    def createNetFromCls(self, clsData): 
        bn = self._createBotHypo(clsData)
        if bn == None:
            return
        else:
            self.botNodes[0] = bn
            self.clauses[0] = clsData
            bot_act_seq = bn.instances[0].getActSeq()
            bot_mapping = bn.instances[0].getMapping()
        self.hypos[len(bn.vard_preds)] = {str(bn):bn}
        
        planner = SymbPlanner()

        if constant.space_pruning == 'prune':
            # each repo here is a list of string, where each string is a conjunction of predicates
            # used to describe the goal state of the training instance. Example string likes:
            # (state Tv_1 IsOn)(Near Robot Tv_1)        
            neg_repo = []
            pos_repo = []
            tmp_pos_repo = []
            tmp_neg_repo = []
            
            pos_repo.append(bn.instances[0].getPredsStr())
            hypoid = 0
            print('hypoid: ')
            while len(pos_repo) != 0:
                tmp_pos_repo = []
                visited_sibs = []
                for hypo_str in pos_repo:
                    sibs = self._getNodeSibsFromStr(hypo_str)    
                    for sib in sibs:
                        hypoid = hypoid+1
                        print(str(hypoid)+" ",end="",flush=True)
                        if hypoid % 100 == 0:
                            print('\n')
                        if sib in visited_sibs:
                            continue
                        elif self._check_could_derive_from_neg_hypo(sib, neg_repo):
                            visited_sibs.append(sib)
                            tmp_neg_repo.append(''.join('('+j+')' for j in sib))
                            continue
                        else:
                            visited_sibs.append(sib)
                        preds = []
                        for pred in sib:
                            p = StatePredicate()
                            p.createFromDataStr(pred)
                            preds.append(p)
                        action_seq = planner.execute_with_initenv_and_goal(
                                                    clsData.getStartEnv(), preds)
                        if self._same_act_seq(action_seq, bot_act_seq):
                            tmp_pos_repo.append(''.join('('+j+')' for j in sib))
                        else:
                            tmp_neg_repo.append(''.join('('+j+')' for j in sib))
                neg_repo = copy.deepcopy(list(set(tmp_neg_repo)))
                tmp_pos_repo = sorted(list(set(tmp_pos_repo)))
                pos_repo = copy.deepcopy(tmp_pos_repo)
                if len(tmp_pos_repo) == 0 or len(tmp_pos_repo[0]) == 0:
                    continue
                else:
                    parent_hypos = self.hypos[min(self.hypos.keys())]
                    hypos = {}
                    from lib.tools import getPredicatesFromStr
                    for conj in tmp_pos_repo: # each conj is a string of conjunctions of grounded predicates
                        pred_str_list = getPredicatesFromStr(conj)
                        effects = []
                        for pred_str in pred_str_list:
                            p = StatePredicate()
                            p.createFromDataStr(pred_str)
                            effects.append(p)
                            
                        objlist = []
                        for effect in effects:
                            objlist = objlist + effect.getPhyObjs()
                        objlist = list(set(objlist))
                        hypo = Hypo()
                        hypo.createFromObjPreds(clsData.dscr, clsData.getArgGroundings(), 
                                                objlist, effects, 
                                                clsData.getStartEnv(), bot_mapping)    
                        hypos[str(hypo)] = hypo
                        for parent_hypo_str in parent_hypos.keys():
                            if hypo.isSibOfConjs(parent_hypo_str):
                                if parent_hypo_str in self.links.keys():
                                    self.links[parent_hypo_str].append(str(hypo))
                                else:
                                    self.links[parent_hypo_str] = [str(hypo)]
                    if len(hypo.vard_preds) == 0:
                        logger.debug('hypo.vard_preds len 0')
                    else:
                        self.hypos[len(hypo.vard_preds)] = hypos
        elif constant.space_pruning == 'nprune':        
            # each repo here is a list of string, where each string is a conjunction of predicates
            # used to describe the goal state of the training instance. Example string likes:
            # (state Tv_1 IsOn)(Near Robot Tv_1)  
            # in the 'nprune' setting, the space keeps all the positive nodes, no pruning involved.      
            pos_repo = []
            tmp_pos_repo = []
            
            pos_repo.append(bn.instances[0].getPredsStr())
            hypoid = 0
            print('hypoid: \n0 ')
            while len(pos_repo) != 0:
                tmp_pos_repo = []
                visited_sibs = []
                for hypo_str in pos_repo:
                    sibs = self._getNodeSibsFromStr(hypo_str)    
                    for sib in sibs:
                        if sib in visited_sibs:
                            continue
                        else:
                            visited_sibs.append(sib)
                        hypoid = hypoid+1
                        print(str(hypoid)+" ",end="",flush=True)
                        if hypoid % 100 == 0:
                            print('\n')
                        preds = []
                        for pred in sib:
                            p = StatePredicate()
                            p.createFromDataStr(pred)
                            preds.append(p)

                        tmp_pos_repo.append(''.join('('+j+')' for j in sib))
            
                tmp_pos_repo = sorted(list(set(tmp_pos_repo)))
                pos_repo = copy.deepcopy(tmp_pos_repo)
                if len(tmp_pos_repo) == 0 or len(tmp_pos_repo[0]) == 0:
                    continue
                else:
                    parent_hypos = self.hypos[min(self.hypos.keys())]
                    hypos = {}
                    from lib.tools import getPredicatesFromStr
                    for conj in tmp_pos_repo: # each conj is a string of conjunctions of grounded predicates
                        pred_str_list = getPredicatesFromStr(conj)
                        effects = []
                        for pred_str in pred_str_list:
                            p = StatePredicate()
                            p.createFromDataStr(pred_str)
                            effects.append(p)
                            
                        objlist = []
                        for effect in effects:
                            objlist = objlist + effect.getPhyObjs()
                        objlist = list(set(objlist))
                        hypo = Hypo()
                        hypo.createFromObjPreds(clsData.dscr, clsData.getArgGroundings(), 
                                                objlist, effects, 
                                                clsData.getStartEnv(), bot_mapping)    
                        hypos[str(hypo)] = hypo
                        for parent_hypo_str in parent_hypos.keys():
                            if hypo.isSibOfConjs(parent_hypo_str):
                                if parent_hypo_str in self.links.keys():
                                    self.links[parent_hypo_str].append(str(hypo))
                                else:
                                    self.links[parent_hypo_str] = [str(hypo)]
                    if len(hypo.vard_preds) == 0:
                        logger.debug('hypo.vard_preds len 0')
                    else:
                        self.hypos[len(hypo.vard_preds)] = hypos
                                  
    def _same_act_seq(self, seq1, seq2):
        if constant.same_seq_crit == 'Crit1':
            if len(seq1) != len(seq2):
                return False
            else:
                for i in range(len(seq1)):
                    inst1 = seq1[i]
                    inst2 = seq2[i]
                    if inst1.instrTheSame(inst2):
                        continue
                    else:
                        return False
            return True                
                
    def _check_could_derive_from_neg_hypo(self, sib, neg_repo):
        from lib.tools import getPredicatesFromStr
        for neg_hypo in neg_repo:
            pred_str_list = getPredicatesFromStr(neg_hypo)
            belong_flag = True
            for i in sib:
                if i not in pred_str_list:
                    belong_flag = False
                    break
                else:
                    continue
            if belong_flag:
                return True
        return False                                    
        
    def _getNodeSibsFromStr(self, predStr):
        from lib.tools import getPredicatesFromStr
        pred_str_list = getPredicatesFromStr(predStr)
        conj_list = []
        if len(pred_str_list) == 1:
            return conj_list
        for pred in pred_str_list:
            conj_list.append([i for i in pred_str_list if i != pred])
        return conj_list

    def _createBotHypo(self, cls):
        logger.debug('CLAUSE INFO: \n'+str(cls))
        clsArgs = cls.getArgGroundings()
        objlist_tobeconsidered = None
        
        """there's a clause 'and wash' in the data, this specific data need to be removed."""
        if cls.sent == 'and wash' and cls.verb == 'wash':
            return None
        
        start_env = cls.getStartEnv()
        end_env = cls.getEndEnv()
        if start_env is None and end_env is None:
            # both envs are None means this clause doesn't have andy action in training data.
            return None
        if constant.node_chosen_heu == "Heuristic1":
            objlist_tobeconsidered = cls.getObjs_argonly()
        elif constant.node_chosen_heu == "Heuristic2":
            objlist_tobeconsidered = cls.getObjs_operated()
        elif constant.node_chosen_heu == "Heuristic3":
            objlist_tobeconsidered = cls.getObjs_hasdirectrelation()
        elif constant.node_chosen_heu == "Heuristic4":
            objlist_tobeconsidered = cls.getObjs_allchanged()
        
        cand_effects = end_env.getObjState(objlist_tobeconsidered)
        neg_cand_effects = start_env.getObjState(objlist_tobeconsidered)
        
        if len(cand_effects) == 0 and len(neg_cand_effects) == 0:
            logger.debug('no state related with the candidate objects.\n================')
            return None        
        else:
            if constant.use_changed_effects:
                changed_effects = start_env.getNonExistencePreds(cand_effects)
                neg_changed_effects_ = end_env.getNonExistencePreds(neg_cand_effects)
                neg_changed_effects = copy.deepcopy(neg_changed_effects_)
                for eff in neg_changed_effects:
                    eff.changeLabelToFalse()
                changed_effects = changed_effects+neg_changed_effects
            else:
                raise Exception('not implemented')#changed_effects = cand_effects
            if len(changed_effects) == 0:
                logger.debug('argument objects have no state change.\n================')
                return None
            elif len(changed_effects) >= 10:
                logger.debug('number of state change larger than 9! Need to abandon.')
            else:
                objlist = []
                for effect in changed_effects:
                    if effect.pred == 'state':
                        objlist.append(effect.getPhyObjs()[0]) # the objlist here is a list of string, 
                                                       # where each string is an object id
                    else:
                        objlist = objlist + list(effect.getPhyObjs())
                objlist = sorted(list(set(objlist)))     
                hypo = Hypo()           
                hypo.createFromObjPreds(cls.dscr, clsArgs, objlist, changed_effects, start_env)  
                if constant.space_pruning == 'prune':          
                    if len(hypo.instances[0].instance_act_seqs) == 0:
                        logger.debug('trying to create bottom node, but the planner cannot generate the action sequence.\n')
                        logger.debug('desired goal state: '+''.join(str(k) for k in changed_effects)+'\n.================')
                        return None
                return hypo                        

    def netMerg(self, hn2):
        # always assume hn2 has only one bottom
        from lib.tools import getPredicatesFromStr
        if constant.space_merg_crit == 'same_node_only':
            raise Exception('space_merg_crit equals same_node_only is not implemented')
        elif constant.space_merg_crit == 'check_links_cross_net':
            # handle the hypos
            for level_key in hn2.hypos.keys():
                if level_key in self.hypos.keys():                    
                    hn1nodes = copy.deepcopy(self.hypos[level_key])
                    hn2nodes = copy.deepcopy(hn2.hypos[level_key])    
                    for node_key in sorted(hn2.hypos[level_key].keys()):
                        if node_key in self.hypos[level_key].keys():
                            hn1nodes[node_key].nodemerge(hn2nodes[node_key])
                        else:
                            hn1nodes[node_key] = copy.deepcopy(hn2nodes[node_key])
                    self.hypos[level_key] = hn1nodes                        
                else:
                    self.hypos[level_key] = copy.deepcopy(hn2.hypos[level_key])
            # handle the links
            tmp_level_nodeid = []
            for level_key in sorted(self.hypos.keys()):
                id_str_list = sorted(list(self.hypos[level_key].keys()))
                temp = []
                for id_str in id_str_list:
                    temp.append(getPredicatesFromStr(id_str))
                tmp_level_nodeid.append(temp)
            tmp_links = {}
            for i in range(0, len(tmp_level_nodeid)-1):
                high_level_ids = tmp_level_nodeid[i]
                low_level_ids = tmp_level_nodeid[i+1]
                for l_id in low_level_ids:
                    for h_id in high_level_ids:
                        if check_h_sibof_l(h_id, l_id):
                            l_id_str = ''.join(['('+i+')' for i in sorted(l_id)])
                            h_id_str = ''.join(['('+i+')' for i in sorted(h_id)])
                            if l_id_str in tmp_links.keys():
                                tmp_links[l_id_str].append(h_id_str)
                            else:
                                tmp_links[l_id_str] = [h_id_str]
                        else:
                            continue
            self.links = copy.deepcopy(tmp_links)
            # handle the botNodes
            if len(hn2.botNodes) == 0:
                pass
            elif len(hn2.botNodes) == 1:
                bot_num = len(self.botNodes)
                self.botNodes[bot_num] = copy.deepcopy(hn2.botNodes[0])
                self.clauses[bot_num] = copy.deepcopy(hn2.clauses[0])
                
        logger.debug('hypo merging with '+hn2.clauses[0].verb+'\n')          
                
                                         
    def get_parents_by_ids(self, node_id_list):
        parents_ids = []
        for parent_id in self.links.keys():
            for child_id in self.links[parent_id]:
                if child_id in node_id_list:
                    parents_ids.append(str(parent_id))
                    break
                else:
                    continue
        return parents_ids
        
    def returnAllHypos(self):
        allhypos_list = []
        for level_id in sorted(self.hypos.keys()):
            hypos = self.hypos[level_id]
            allhypos_list = allhypos_list + list([hypos[k] for k in sorted(hypos.keys())])
        return allhypos_list
            
    def getSize(self):
        return sum([len(k) for k in self.hypos.values()])   
   
        
class Hypo(object):
    def __init__(self):
        self.freq = 0
        self.vard_preds = []
        self.var_affords = {}
        self.instances = []
        
    def getNodeFreq(self):
        return self.freq
    
    def nodemerge(self, node2):
        self.freq = self.freq+1
        self.instances = self.instances+node2.instances
        
    def createFromObjPreds(self, dscr, args, objlist, preds, start_env, mapping = None):            
        self.freq = self.freq + 1
        hi = HypoInstance()   
        hi.dscr = dscr
        hi.instance_args = args  
        hi.instance_preds = preds
        hi.instance_objs = objlist
        hi.start_env = start_env
        planner = SymbPlanner()
        if constant.space_pruning == 'prune':
            hi.instance_act_seqs = planner.execute_with_initenv_and_goal(start_env, preds)
        if mapping == None:
            hi.instance_var_mapping = self._formMapping(args, objlist)
        else:
            hi.instance_var_mapping = mapping
        self.instances.append(hi)       
        self.vard_preds = self._gen_vard_preds(preds, hi.instance_var_mapping) 
        self.var_affords = self._gen_var_affords(start_env.getObj(), hi.instance_var_mapping) 
        self._get_id()

    def isSibOfConjs(self, parent_conj_str):
        from lib.tools import getPredicatesFromStr
        self_preds = getPredicatesFromStr(self.id)
        parent_preds = getPredicatesFromStr(parent_conj_str)
        if check_h_sibof_l(self_preds, parent_preds):
            return True
        else:
            return False


    def _formMapping(self, args, objlist):
        map = {}
        id = 0
        for key in sorted(args.keys()):
            map[args[key]] = key
            id = id+1
        id = 0
        for obj in objlist:
            if obj in map.keys():
                continue
            else:
                map[obj] = 'obj'+str(id)
                id = id+1
        return map        

    def _gen_vard_preds(self, preds, map):
        vard_preds = []
        for pred in preds:
            tmppred = copy.deepcopy(pred)
            for key,value in pred.args.items():
                if str(value) in map.keys():
                    tmppred.args[key] = map[str(value)]
            vard_preds.append(copy.deepcopy(tmppred))
        return vard_preds      
    
    def _gen_var_affords(self, objs, map):
        var_affords = {}
        for obj in objs:
            obj_id = obj.idfer
            if obj_id in map.keys():
                if obj_id == 'robot':
                    var_affords[map[obj_id]] = ['isarobot']
                else:
                    var_affords[map[obj_id]] = obj.getObjAfford()
        return var_affords   
    
    def _get_id(self):
        tmp_map = {}
        for var in self.var_affords.keys():
#             if self.var_affords[var] == ['isarobot']:
#                 tmp_map[str(var)] = 'robot'
#             else:
#                 tmp_map[str(var)] = str(var)
            if str(var).startswith('var'):
                tmp_map[str(var)] = str(var)
            else:
                tmp_map[str(var)] = '['+'^'.join(self.var_affords[var])+']'
        self.id = ''
        preds = []
        for pred in self.vard_preds:
            predstr = pred.getPredStr()
            for i in tmp_map.keys():
                predstr = predstr.replace(str(i), tmp_map[i])
            preds.append(predstr)
        self.id = ''.join(k for k in sorted(preds))
    
    def __str__(self):
        #return ''.join(pred.getPredStr() for pred in self.vard_preds)
        return self.id
        
class HypoInstance(object):
    def __init__(self):
        self.instance_preds = [] #the goal state predicates from the training instance
        self.instance_objs = [] #the objects from the goal state
        self.instance_args = {} #the arguments of the trainingaction frame
        self.instance_act_seqs = {} #the corresponding action sequence to achieve the goal state from the start env
        self.instance_var_mapping = {}   
        self.start_env = None       
        self.dscr = '' 

    def getPredsStr(self):
        return ''.join(sorted([pred.getPredStr() for pred in self.instance_preds]))      
    
    def getActSeq(self):
        return self.instance_act_seqs  

    def getMapping(self):
        return self.instance_var_mapping
    
    def __str__(self):
        toreturn = '        predicates:  '+'^'.join(sorted([str(p) for p in self.instance_preds]))+'\n'+\
                   '        var and obj: '+'('+' '.join(sorted([ str(self.instance_var_mapping[k])+':'+str(k) for k in self.instance_var_mapping.keys()]))+')'+'\n'
        return toreturn
import copy

import constant
from expression import Expression

class Simulator(object):
    
    def __init__(self):
        self.preConditions = None # preCondition[ [string, [str, str, str], Expression],
                                  #               [string, [str, str, str], Expression],
                                  #               [string, [str, str, str], Expression]]
        self.effect = None # effect        [ [string, [str, str, str], Expression],
                           #               [string, [str, str, str], Expression],
                           #               [string, [str, str, str], Expression]]
    
    def formSimulator(self):
        # Initializes the preConditions and effect by reading the domainKnowledge.pddl file
        self.preConditions = []
        self.effect = []
        
        lines = None
        fid = open(constant.domain_file, 'r')
        lines = fid.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].lower()
        fid.close()
        
        actionName = ''
        variables = []
        
        i=0
        while i < len(lines):
            lines[i] = lines[i].lower()
            if lines[i].startswith('(:action'):
                tmp = lines[i][len('(:action'):]
                col = tmp.index(':')
                parenOpen = tmp.index('(')
                parenClose = tmp.index(')')
                actionName = tmp[:col].strip()
                variables = tmp[parenOpen+1 : parenClose].split(' ')
                if len(variables[0]) == 0:
                    variables.remove('')
            
            if lines[i].startswith(':precondition'):
                precondition = ''
                while (not lines[i].startswith(':effect')):
                    
                    precondition = precondition+lines[i]
                    i = i+1
                exp = Expression()
                exp.formExpression(precondition)
                self.preConditions.append([actionName, variables, exp])
            
            if lines[i].startswith(':effect'):
                effect = ''
                while ( i < len(lines) and len(lines[i].strip())!=0 ):
                    effect = effect + lines[i]
                    i = i+1
                exp = Expression()
                exp.formExpression(effect)
                self.effect.append([actionName, variables, exp])
            
            i = i + 1
        # special cases
        # some actions may have empty preconditions or effect
        # which are not written in the planner, we handle them here
        exp_prewait = Expression()
        exp_prewait.formExpression('()')
        exp_postwait = Expression()
        exp_postwait.formExpression('()')
        exp_premoveto = Expression()
        exp_premoveto.formExpression('()')
        self.preConditions.append(['wait',[],exp_prewait])
        self.effect.append(['wait',[],exp_postwait])
        self.preConditions.append(['moveto',['?o'],exp_premoveto])
            
            
    def getInstructionAlias(self, inst):
        cf = inst.getInstVerb()
        dscp = inst.getInstArgs()
        
        if (cf == 'press' or cf == 'open' or cf == 'close' or cf == 'turn'):
            return [cf+'_'+dscp[0], []]
        elif (cf == 'add' or cf == 'place'):
            return [cf+'_'+dscp[0], [dscp[1]]]
        elif (cf == 'keep'):
            for pc in self.preConditions:
                if pc[0].startswith('keep'):
                    first_ = pc[0].find('_')
                    if first_ == -1:
                        break
                    second_ = pc[0][first_+1:].find('_')
                    if second_ == -1:
                        break
                    second_ = second_+first_+1
                    first = pc[0][first_+1 : second_]
                    second = pc[0][second_+1:]
                    if (first == dscp[1] and second == dscp[2]):
                        return [pc[0], [dscp[0]]]
        return [cf, [dscp[k] for k in sorted(dscp.keys())]]

    def execute(self, inst, env, force=True, copyenv=True):
        """ inst is an Instruction()
            env is and Environment()
        """
        # force is always True, so here we never check the precondition
        
        present = None
        if (copyenv):
            present = copy.deepcopy(env)
        else:
            present = env
        alias = self.getInstructionAlias(inst)
        found = False
        for eff in self.effect:
            if (eff[0].lower() == alias[0].lower()):
                map = [] # list of list, example: [['?o', 'mug_1'],['?x', 'stove']]
                if len(alias[1]) != len(eff[1]):
                    continue
                found = True
                for i in range(len(eff[1])):
                    map.append([ eff[1][i], alias[1][i] ])
            
                eff[2].modify(present, map)
                break
        if not found:
            raise Exception('Cannot parse the instruction: '+str(inst))
            
        return present
    
    def executeList(self, instlist, env, force=True):
        copy = copy.deepcopy(env)
        for inst in instlist:
            copy = self.execute(inst, copy, force)
        return copy
            
            
            
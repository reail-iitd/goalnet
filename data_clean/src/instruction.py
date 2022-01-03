class Instruction(object):
    def __init__(self):
        self.verb = '' # self.verb is a string
        self.args = {} # self.args is a dictrionary of string
        
    def instrTheSame(self, instr):
        if str(instr) != self.__str__():
            return False
        else:
            return True
        
    def getInstVerb(self):
        return self.verb
    
    def getInstArgs(self):
        return self.args
    
    def getInstArgObjs(self):
        if len(self.args) == 1:
            return list(self.args.values())
        elif len(self.args) == 2:
            return list(self.args.values())
        elif len(self.args) == 3:
            return [self.args[0], self.args[2]]
        elif len(self.args) == 0:
            return []
        else:
            raise Exception('instruction has more than 3 arguments')
        
    def createInstFromDataStr(self, inststr):
        self.verb = inststr.split(' ')[0]
        if len(inststr.split(' ')) == 1:
            pass
        else:
            args = inststr.split(' ')[1:] 
            for i in range(len(args)):
                self.args[i] = args[i]
    
    def createInstFromPlanResult(self, inststr):
        inststr = inststr.replace(' ','')   
        # The following rules are designed based on the domainknowledge.pddl
        # Each rule corresponds to a specific type of primitive action.
        if inststr.index("(") == inststr.index(")")-1:
            #the instruction doesn't have parameter, and the first "_" in the action is to separate the verb and argument
            #example instruction: press_MicrowaveButton
            self.verb = inststr.split("_")[0]
            self.args[0] = inststr[inststr.find("_")+1 : inststr.index("(")]
            
        elif inststr[:inststr.find('(')].count('_') == 0:
            #pred part has no "_", and all arguments are between ()
            #Example: grasp(***)
            self.verb = inststr.split('(')[0]
            args = inststr[inststr.find('(')+1 : inststr.find(')')].split(',')
            for i in range(len(args)):
                self.args[i] = args[i]        
        
        elif inststr.startswith('keep_'):
            #This type of instruction has only one parameter, and the pred part uses 'keep_'
            #Example: keep_in_garbagebin_1(***)
            self.args[0] = inststr[inststr.find('(')+1 : inststr.find(')')]
            str_elem = inststr[:inststr.find('(')].split('_')
            self.verb = str_elem[0]
            self.args[1] = str_elem[1]
            self.args[2] = '_'.join(i for i in str_elem[2:])        
        
        elif inststr.startswith('add_'):
            self.verb = inststr.split('_')[0]
            self.args[0] = inststr[inststr.find('_')+1:inststr.find('(')]
            self.args[1] = inststr[inststr.find('(')+1:inststr.find(')')]

        elif inststr.startswith('place_'):
            self.verb = inststr.split('_')[0]
            self.args[0] = inststr[inststr.find('_')+1:inststr.find('(')]
            self.args[1] = inststr[inststr.find('(')+1:inststr.find(')')]

        else:
            raise Exception(inststr)

    def __str__(self):
        return self.getInstVerb()+'('+' '.join([self.args[key] for key in sorted(self.args.keys())])+')' 
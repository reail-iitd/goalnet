class StatePredicate(object):
    """description of class"""
    def __init__(self):
        self.pred = ''
        self.args = {}
        self.tf_label = True
   
    def changeobj(self, obj_map):
        #obj_map is a mapping of old obj to new obj: {old:new, old:new}
        for k in self.args.keys():
            if self.args[k] in obj_map.keys():
                self.args[k] = obj_map[self.args[k]]
    
    def getPred(self):
        return self.pred.lower()

    def getArgs(self):
        return self.args

    def getTfLabel(self):
        return self.tf_label

    def changeLabelToFalse(self):
        self.tf_label = False

    def createFromDataStr(self, predstr):
        predstr = predstr.lower()
        if predstr[0] == '(' and predstr[-1] == ')':
            raise Exception('createFromDataStr starts with ( and ends with ) ')
        if predstr.startswith('not(') and predstr.endswith(')'):
            self.changeLabelToFalse()
            predstr = predstr[4:-1]
        
        self.pred = predstr.split(' ')[0]
        args = predstr.split(' ')[1:]
        for i in range(len(args)):
            self.args[i] = args[i].lower()

    def getPredStr(self):
        if self.tf_label:
            tmp = "("+self.pred+" "+" ".join([self.args[k] for k in sorted(self.args.keys())])+")"
        else:
            tmp = "(not("+self.pred+" "+" ".join([self.args[k] for k in sorted(self.args.keys())])+"))"
        return tmp.lower()
    
    def getPhyObjs(self):
        if self.pred == 'state':
            return [self.args[0]]
        else:
            return list(self.args.values())
    def __str__(self):
        return self.getPredStr()


import copy
from statepredicate import StatePredicate

class Environment(object):
    def __init__(self):
        self.objlist = []
        self.predlist = []
        
    def createObjList(self, objlist):
        self.objlist = copy.deepcopy(objlist)
        
    def createPredListFromDataStr(self, predliststr):
        for predstr in predliststr.split(','):
            predstr = predstr.strip()
            pred = StatePredicate()
            pred.createFromDataStr(predstr[1:-1])
            self.predlist.append(pred)
            
    def getObjState(self, objlist):
        predtoreturn = []
        for obj in objlist:
            for pred in self.predlist:
                if pred.getPred() in ['isgraspable', 'issqueezeable']:
                    continue
                if obj in pred.getPhyObjs():
                    if ''.join([str(p) for p in predtoreturn]).find(str(pred)) == -1:
                        predtoreturn.append(pred)
        return copy.deepcopy(predtoreturn)
    
    def getNonExistencePreds(self, predlist):
        nonExistencePreds = []
        for pred in predlist:
            if not pred.getPredStr().startswith('(not('):
                if any(tmppred.getPredStr()==pred.getPredStr() for tmppred in self.predlist):
                    continue
                else:
                    nonExistencePreds.append(pred)
            else:
                if any(tmppred.getPredStr()==pred.getPredStr()[4:-1] for tmppred in self.predlist):
                    nonExistencePreds.append(pred)
                else:
                    continue
        return nonExistencePreds
    
    def getObj(self):
        return self.objlist
    
    def getObjsStr(self):
        objsStr = ""
        for obj in self.objlist:
            objsStr = objsStr + " " + obj.getObjName()
        return objsStr
    
    def checkPreStrExist(self, constraint):
        if constraint[0] == '(' and constraint[-1] == ')':
            constraint = constraint[1:-1]
        for pred in self.predlist:
            if pred.getPredStr()[1:-1] == constraint:
                return True
        return False               
                    
    def getAllPreds(self):
        return self.predlist    
    
    def getPredsStr(self):
        return " ".join(sorted([i.getPredStr() for i in self.predlist]))    
    
    def getAffordsByObjId(self,objId):
        if isinstance(objId, str):
            for obj in self.objlist:
                if obj.getObjName() == objId:
                    return obj.getObjAfford()
        else:
            raise Exception('environment getAffordsByObjId does not support \
                            group objects yet.')    
            
    def findObject(self, objId):
        for obj in self.objlist:
            if obj.getObjName() == objId:
                return obj
        return None 
    
    def modify(self, constraint, truth):
        # constraint is a string, truth is a bool value
        """Constraint represents a relation which is of one of the two types:
           1. (state objName stateName) or 2. (relation objName1 objName2)
           If true is true then make these conditions true else make them false
        """
        words = constraint.split(' ')
        if words[0] == 'state':
            objFound = self.findObject(words[1])
            if objFound is None:
                #raise Exception('obj in constraint is not in the evn')
                return
            if truth:
                pred = StatePredicate()
                pred.createFromDataStr(constraint)
                self.predlist.append(pred)
            else:
                for i in range(len(self.predlist)):
                    if self.predlist[i].getPredStr()[1:-1] == constraint:
                        del self.predlist[i]
                        return
                # raise Exception('obj in constraint in the env, but the state is not in')
        elif len(words) == 3:
            if truth:
                pred = StatePredicate()
                pred.createFromDataStr(constraint)
                self.predlist.append(pred)
            else:
                for i in range(len(self.predlist)):
                    if self.predlist[i].getPredStr()[1:-1] == constraint:
                        del self.predlist[i]
                        return
                
                
        else:
            raise Exception('PDDL-Constraint Parser Error: Cannot Parse Constraint - '+constraint+'.')  
        
    def isSatisfied(self, constraint):
        if constraint == 'true' or constraint == '':
            return 1
        words = constraint.split(' ')
        if words[0] == 'state':
            objFound = self.findObject(words[1])
            if objFound == None:
                return -1
            if self.checkPreStrExist(constraint):
                return 1
            else:
                return 0
        elif len(words) == 2:
            objFound = self.findObject(words[1])
            if objFound is None or words[0] in objFound.getObjAfford():
                return -1
            return 1
        elif len(words) == 3:
            if words[0] == '=':
                if words[1] == words[2]:
                    return 1
                else:
                    return 0
            
            obj1 = self.findObject(words[1])
            obj2 = self.findObject(words[2])
            if obj1 is None or obj2 is None:
                return -1
            if self.checkPreStrExist(constraint):
                return 1
            else:
                return 0
        else:
            raise Exception('PDDL-Constraint Parser Error: Cannot Parse Constraint - '+constraint+'.')   
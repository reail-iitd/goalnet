class PhysicalObject(object):
    """description of class"""
    def __init__(self):
        self.idfer = ''
        self.affordances = []

    def buildFromStr(self,rawspstring):
        spstring = rawspstring[1:-1]
        self.idfer = spstring.split(':')[0].lower()
        if len(spstring.split(':')[1]) == 0:
            pass
        else:
            self.affordances = list(set([i.lower() for i in spstring.split(':')[1].split(',')]))

    def getObjName(self):
        return self.idfer

    def getObjAfford(self):
        if self.idfer == 'robot':
            self.affordances.append('isarobot')
        return self.affordances
    



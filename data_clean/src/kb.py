import copy
import logging

import constant

logger = logging.getLogger(__name__)

def get_verb_frame(cls):
    kb = VerbSemKB()
    frame_str = kb._get_verb_strc_str(cls)
    return frame_str

class VerbSemKB(object):
    """description of class"""
    def __init__(self):
        # The terms here is a dictionary of dictionary. 
        # For example: {verb1:{verb1_structure1:net, verb1_structure2: net}
        #               verb2:{verb2_structure1:net, verb2_structure2: net}
        #               }
        self.terms = {}
        
    def checkVerbStrcExists(self, cls):
        verb = self._get_verb_from_cls(cls)
        verb_strc_str = self._get_verb_strc_str(cls)
        if verb in self.terms.keys():
            if verb_strc_str in self.terms[verb].keys():
                return True
            else:
                return False
        else:
            return False
    
    def checkVerbExists(self, cls):
        # check whether certain verb already exists in the keys of the knowledge base.
        if self._get_verb_from_cls(cls) in self.terms.keys():
            return True
        else:
            return False

    def createTerm(self, cls, hyponet):
        # the verb in the cls doesn't exist in the knowledge base. 
        # So here we will create a term in the kb with the verb as the key. And 
        # create a verb structure under this verb, with the verb_strc_str as the key.
        verb = self._get_verb_from_cls(cls)
        self.terms[verb] = {}
        verb_strc_str = self._get_verb_strc_str(cls)
        self.terms[verb][verb_strc_str] = copy.deepcopy(hyponet)
        logger.debug('create verb term for: '+self._get_verb_from_cls(cls)+'\n')
        logger.debug('create action structure for: '+self._get_verb_strc_str(cls)+'\n================')

    def createStrc(self, cls, hyponet):
        verb = self._get_verb_from_cls(cls)
        verb_strc_str = self._get_verb_strc_str(cls)
        self.terms[verb][verb_strc_str] = copy.deepcopy(hyponet)
        logger.debug('create action structure for: '+self._get_verb_strc_str(cls)+'\n================')

    def mergeNet(self, cls, hyponet):
        verb = self._get_verb_from_cls(cls)
        verb_strc_str = self._get_verb_strc_str(cls)
        self.terms[verb][verb_strc_str].netMerg(hyponet)
        logger.debug('merge net for structure: '+self._get_verb_strc_str(cls)+'\n================')
    
    def getHypoNetForCls(self, cls):
        return self.terms[self._get_verb_from_cls(cls)][self._get_verb_strc_str(cls)]

    def _get_verb_from_cls(self, cls):
        # return the verb of the clause
        return cls.getVerb()

    def _get_verb_strc_str(self, cls):
        # form a string representation for the action frame.
        # different criterions to form this representation could be found in constant.py
        verb_strc_str = ''
        if constant.same_verbstrc_crit == 'same_verb_same_argnumb':
            verb_strc_str = self._get_verb_from_cls(cls)
            var_id = 0
            for key in sorted(cls.getArgs().keys()):
                if cls.getArgs()[key] in cls.getRefinedArgsGroundings().keys():
                    verb_strc_str = verb_strc_str + '+var'+str(var_id)
                    var_id = var_id+1
            return verb_strc_str


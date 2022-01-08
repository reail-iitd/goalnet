import logging
import os
import pickle
import copy

import constant
from simulator import Simulator
from clausedata import ListClauseData
from inducer import VerbSemInducer
from evl import Evaluator
from inference import Inferencer


class Data(object):
    def __init__(self):
        self.data_bycls = []
        self.data_byfile = []
        
        self._formData()
        
    def _formData(self):
        if constant.use_precomputed_data:
            self.data_bycls = pickle.load(open(constant.precomputed_dataset_dir+constant.precomputed_clausedata_fname,"rb"))
            self.data_byfile = pickle.load(open(constant.precomputed_dataset_dir+constant.precomputed_filedata_fname,"rb"))
        else:
            self._rawDataExtraction()
        
    def _rawDataExtraction(self):
        """pre-process the raw data"""
        datafiles = os.listdir(constant.data_dir)
        if len(datafiles) % 2 == 0:
            pass
        else:
            raise Exception("data comes with clause:evn pairs, #data should be even!")
        for index in range(int(len(datafiles)/2)):
            clsfname = str(index)+'.clauses'
            envfname = str(index)+'.instenv'
            lclsdata = ListClauseData(constant.data_dir, clsfname, envfname)
            if len(lclsdata.returnClauses()) > 0:
                self.data_bycls = self.data_bycls + lclsdata.returnClauses()
                self.data_byfile.append(lclsdata)
        pickle.dump(self.data_bycls, open(constant.precomputed_dataset_dir+constant.precomputed_clausedata_fname, 'wb'))
        pickle.dump(self.data_byfile, open(constant.precomputed_dataset_dir+constant.precomputed_filedata_fname, 'wb'))    
        constant.dump_planning_result()
    def getClsData(self):
        return self.data_bycls
    def getFileData(self):
        return self.data_byfile                
        
class DataSegConfig(object):
    def __init__(self, train_set = [], test_set = [], dev_set = []):
        self.train_set = train_set
        self.test_set = test_set
        self.dev_set = dev_set
    def assignTrainSet(self, train):
        self.train_set = train
    def assignTestSet(self, test):
        self.test_set = test
    def assignDevSet(self, dev):
        self.dev_set = dev
    def getTrainSet(self):
        return self.train_set       
    def getTestSet(self):
        return self.test_set
    def getDevSet(self):
        return self.dev_set


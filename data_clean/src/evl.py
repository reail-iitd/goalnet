import constant
import logging
#import copy

logger = logging.getLogger(__name__)

class Evaluator(object):
    def __init__(self):
        pass
    
    def evl_pack(self, gt, predicted):
        """both gt and predict_result are sequence of instenvpair"""
        evl_result = ResultMatrix()
        
        # calculate edit distance

        evl_result.seq_edit_distance_score = self.evl_distance(gt, predicted)
        
        # calculate jaccard index
        evl_result.final_env_diff_jac_score = self.evl_jacindex(gt, predicted)

        logger.debug('edit '+'{0:.4f}'.format(evl_result.getEdit())+'\n==============')
        logger.debug('jac '+'{0:.4f}'.format(evl_result.getJac())+'\n==============')
        return evl_result
    
    def evl_distance(self, gt, predicted):
        """both gt and predict_result are sequence of instenvpair"""

        # calculate edit distance
        gt_inst_seq = {}
        for k in sorted(gt.keys()):
            if gt[k].getInst() is not None:
                gt_inst_seq[len(gt_inst_seq)] = gt[k].getInst()
        
        inferred_inst_seq = {}
        for k in sorted(predicted.keys()):
            if predicted[k].getInst() is not None:
                inferred_inst_seq[len(inferred_inst_seq)] = predicted[k].getInst()
            
        # seq_edit_distance_score = 1 - self.levenshtein(gt_inst_seq, inferred_inst_seq) / (max(len(gt_inst_seq),len(inferred_inst_seq))+constant.epsilon)
        seq_edit_distance_score = self.calculate_jac_index(gt_inst_seq, inferred_inst_seq)
        return seq_edit_distance_score

    def evl_jacindex(self, gt, predicted):
        min_id = min(gt.keys())
        max_id = max(gt.keys())
        pos_gt_final_env_diff = gt[min_id].getEnv().getNonExistencePreds(gt[max_id].getEnv().getAllPreds())
        neg_gt_final_env_diff = gt[max_id].getEnv().getNonExistencePreds(gt[min_id].getEnv().getAllPreds())
        
        if len(predicted) == 0:
            pos_predict_final_env_diff = []
            neg_predict_final_env_diff = []
        else:
            min_id = min(predicted.keys())
            max_id = max(predicted.keys())
            pos_predict_final_env_diff = predicted[min_id].getEnv().getNonExistencePreds(predicted[max_id].getEnv().getAllPreds())
            neg_predict_final_env_diff = predicted[max_id].getEnv().getNonExistencePreds(predicted[min_id].getEnv().getAllPreds())
        final_env_diff_jac_score = self._jaccard_index(\
                        pos_gt_final_env_diff, pos_predict_final_env_diff, \
                        neg_gt_final_env_diff, neg_predict_final_env_diff)   
        return final_env_diff_jac_score
                                                        
    def levenshtein(self, instseq1, instseq2):
        return self._levenshtein(instseq1, instseq2)
    def _levenshtein(self, instseq1, instseq2):
        m = len(instseq1)
        n = len(instseq2)
        if min(m,n) == 0:
            return max(m,n)
        cost_matrix = {}
        for i in range(m+1):
            for j in range(n+1):
                if min(i,j) == 0:
                    cost_matrix[str(i)+'_'+str(j)] = max(i,j)
                    continue
                cost = 1
                try:
                    if instseq1[i-1].instrTheSame(instseq2[j-1]):
                        cost = 0
                except:
                    raise Exception('levenshtein calculation error')
                a = cost_matrix[str(i-1)+'_'+str(j)]+1
                b = cost_matrix[str(i)+'_'+str(j-1)]+1
                c = cost_matrix[str(i-1)+'_'+str(j-1)]+cost
                cost_matrix[str(i)+'_'+str(j)] = min(a, min(b,c))
        return float(cost_matrix[str(m)+'_'+str(n)])
    
    def calculate_jac_index(self, instseq1, instseq2):
        union = len(instseq1)+len(instseq2)
        inter = 0
        for s in instseq1:
            if s in instseq2:
                inter+=1
                union-=1
        jac = inter *1.0/union
        return jac

    def _jaccard_index(self, pos_predset1, pos_predset2, neg_predset1, neg_predset2):
        pos_predset1 = list(set(pos_predset1))
        pos_predset2 = list(set(pos_predset2))
        neg_predset1 = list(set(neg_predset1))
        neg_predset2 = list(set(neg_predset2))
        intersection = []
        if len(pos_predset2) != 0:
            pos_predset2_str = ''.join([str(pred) for pred in pos_predset2])
            for p1 in pos_predset1:
                if pos_predset2_str.find(str(p1)) != -1:
                    intersection.append(p1)
        if len(neg_predset2) != 0:
            neg_predset2_str = ''.join([str(pred) for pred in neg_predset2])
            for p1 in neg_predset1:
                if neg_predset2_str.find(str(p1)) != -1:
                    intersection.append(p1)
        
        return len(intersection) / (len(pos_predset1)+len(pos_predset2)+ \
                                    len(neg_predset1)+len(neg_predset2)- \
                                    len(intersection)+constant.epsilon)


class ResultMatrix(object):
    def __init__(self):
        self.seq_edit_distance_score = 0
        self.final_env_diff_jac_score = 0
    def getEdit(self):
        return self.seq_edit_distance_score
    def getJac(self):
        return self.final_env_diff_jac_score
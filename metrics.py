import sklearn.metrics as metrics
import numpy as np

def mape(y_true, y_pred):
    return np.minimum(np.abs((y_true - y_pred) / (y_true + 1e-10) * 2), np.ones_like(y_true)) * 100

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

class meter_score_metrics:
    def __init__(self, time_y=[15, 30, 45]):
        self.time_y = time_y
        self.time_len = len(time_y)
        self.acc_scores = []
        self.f1_scores = []
        '''
            scores = [score_15min, score_30min, score_45min]
            score = [acc, f1]
        '''
        
    def update(self, y_true, y_pred):
        y_true = y_true.view(-1, self.time_len).detach().cpu().bool()
        y_pred = y_pred.view(-1, self.time_len).detach().cpu() > 0.5
        this_acc = np.empty(self.time_len)
        this_f1 = np.empty(self.time_len)
        for t in range(self.time_len):
            this_acc[t] = metrics.accuracy_score(y_true[:, t], y_pred[:, t])
            # self.scores[1] += metrics.precision_score(y_true, y_pred)
            # self.scores[2] += metrics.recall_score(y_true, y_pred)
            this_f1[t] = metrics.f1_score(y_true[:, t], y_pred[:, t])
        
        self.acc_scores.append(this_acc)
        self.f1_scores.append(this_f1)
        
    def score_np(self):
        return np.stack(self.acc_scores), np.stack(self.f1_scores)
    
    def score_dict(self):
        acc_res = np.stack(self.acc_scores).mean(axis=0)
        f1_res = np.stack(self.f1_scores).mean(axis=0)
        res_dict = {}
        for i, min in enumerate(self.time_y):
            res_dict['me_acc_%dmin'%min] = acc_res[i]
            res_dict['me_f1_%dmin'%min] = f1_res[i]
        return res_dict
    
class garage_score_metrics:
    def __init__(self, time_y=[15, 30, 45]):
        self.time_y = time_y
        self.time_len = len(time_y)
        self.mae_scores = []
        self.mape_scores = []
        '''
            scores = [score_15min, score_30min, score_45min]
            score = [mae, mape]
        '''
    
    def update(self, y_true, y_pred):
        y_true = y_true.view(-1, self.time_len).detach().cpu()
        y_pred = y_pred.view(-1, self.time_len).detach().cpu()
        
        self.mae_scores.append(mae(y_true, y_pred))
        self.mape_scores.append(mape(y_true, y_pred))
        
    def score_np(self):
        return np.stack(self.mae_scores), np.stack(self.mape_scores)
    
    def score_dict(self):
        mae_res = np.stack(self.mae_scores).mean(axis=0).mean(axis=0)
        mape_res = np.stack(self.mape_scores).mean(axis=0).mean(axis=0)
        res_dict = {}
        for i, min in enumerate(self.time_y):
            res_dict['ga_mae_%dmin'%min] = mae_res[i]
            res_dict['ga_mape_%dmin'%min] = mape_res[i]
        return res_dict
    
def generate_score_metrics(time_y=[15, 30, 45], dtype='meter'):
    if dtype == 'meter':
        return meter_score_metrics(time_y)
    elif dtype == 'garage':
        return garage_score_metrics(time_y)
    elif dtype == 'combine':
        return [garage_score_metrics(time_y), meter_score_metrics(time_y)]
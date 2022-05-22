import sklearn.metrics as metrics
import numpy as np

def mape(y_true, y_pred):
    return np.minimum(np.abs((y_true - y_pred) / (y_true + 1e-10) * 2), np.ones_like(y_true)).mean() * 100

class meter_score_metrics:
    def __init__(self, time_y=[15, 30, 45]):
        self.time_y = time_y
        self.time_len = len(time_y)
        self.scores = np.zeros((self.time_len, 2))
        '''
            scores = [score_15min, score_30min, score_45min]
            score = [acc, f1]
        '''
        self.count = 0
        
    def update(self, y_true, y_pred):
        try:
            y_true = y_true.view(-1, self.time_len).cpu().detach().bool()
            y_pred = y_pred.view(-1, self.time_len).cpu().detach() > 0.5
            self.count += 1
            for t in range(self.time_len):
                self.scores[t, 0] += metrics.accuracy_score(y_true[:, t], y_pred[:, t])
                # self.scores[1] += metrics.precision_score(y_true, y_pred)
                # self.scores[2] += metrics.recall_score(y_true, y_pred)
                self.scores[t, 1] += metrics.f1_score(y_true[:, t], y_pred[:, t])
        except Exception as e:
            print(e)
            
        
    def score_np(self):
        return self.score_np / self.count
    
    def score_dict(self):
        res = self.scores / self.count
        res_dict = {}
        for i, min in enumerate(self.time_y):
            res_dict['me_acc_%dmin'%min] = res[i, 0]
            res_dict['me_f1_%dmin'%min] = res[i, 1]
        return res_dict
    
class garage_score_metrics:
    def __init__(self, time_y=[15, 30, 45]):
        self.time_y = time_y
        self.time_len = len(time_y)
        self.scores = np.zeros((self.time_len, 2))
        '''
            scores = [score_15min, score_30min, score_45min]
            score = [mae, mape]
        '''
        self.count = 0
    
    def update(self, y_true, y_pred):
        try:
            y_true = y_true.view(-1, self.time_len).cpu().detach()
            y_pred = y_pred.view(-1, self.time_len).cpu().detach()
            self.count += 1
            for t in range(self.time_len):
                self.scores[t, 0] += metrics.mean_absolute_error(y_true[:, t], y_pred[:, t])
                # self.scores[1] += metrics.precision_score(y_true, y_pred)
                # self.scores[2] += metrics.recall_score(y_true, y_pred)
                self.scores[t, 1] += mape(y_true[:, t], y_pred[:, t]) ##上界100
        except Exception as e:
            print(e)
    
    def score_np(self):
        return self.score_np / self.count
    
    def score_dict(self):
        res = self.scores / self.count
        res_dict = {}
        for i, min in enumerate(self.time_y):
            res_dict['ga_mae_%dmin'%min] = res[i, 0]
            res_dict['ga_mape_%dmin'%min] = res[i, 1]
        return res_dict
    
    
def generate_score_metrics(time_y=[15, 30, 45], dtype='meter'):
    if dtype == 'meter':
        return meter_score_metrics(time_y)
    elif dtype == 'garage':
        return garage_score_metrics(time_y)
    elif dtype == 'combine':
        return [garage_score_metrics(time_y), meter_score_metrics(time_y)]
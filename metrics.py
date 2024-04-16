import numpy as np

class My_metrics():
    def __init__(self, metrics=['accs', 'precisions', 'recalls', 'f1s', 'aucs', 'aprs'], folds=10):
        self.metrics = {}
        self.folds = folds
        for metric in metrics:
            self.metrics[metric] = {
                f'fold{i+1}': [] for i in range(folds)
            }

    def get_fold_best(self, metric, fold):
        return max(self.metrics[metric][fold]), self.metrics[metric][fold].index(max(self.metrics[metric][fold]))

    def get_item(self, metric, fold, idx):
        return self.metrics[metric][fold][idx]
    
    def get_final(self, metric):
        return np.mean([max(self.metrics[metric][f'fold{i+1}']) for i in range(self.folds)]), np.std([max(self.metrics[metric][f'fold{i+1}']) for i in range(self.folds)])
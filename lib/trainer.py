import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt

TRAIN, TEST, VAL = 0,1,2

class Trainer:
    
    def __init__(self, coherence_matrix_train, coherence_matrix_test, 
                       split_mask,
                       input_features, feature_to_predict, fit_log=False):
        
        self.coherence_matrix_train = coherence_matrix_train
        self.coherence_matrix_test = coherence_matrix_test
        self.split_mask = split_mask
        self.input_features = input_features
        self.feature_to_predict = feature_to_predict
        self.fit_log = fit_log

        self.cmmap = {'Shh':    (0,0), 'ShhShv': (0,1), 'ShhSvv': (0,2),
                      'ShvShh': (1,0), 'Shv':    (1,1), 'ShvSvv': (1,2),
                      'SvvShh': (2,0), 'SvvShv': (2,1), 'Svv':    (2,2)}

        if not (coherence_matrix_test.shape[:2] == coherence_matrix_train.shape[:2]):
            raise ValueError("both coherence matrices must have the same pixel size")

        if not (coherence_matrix_train.shape[2:] == coherence_matrix_train.shape[2:] == (3,3)):
            raise ValueError("coherence matrices must be of shape [h,w,3,3]")

        if sum([i in self.cmmap.keys() for i in input_features])!=len(input_features):
            raise ValueError(f"invalid input features {input_features}")

        if not feature_to_predict in self.cmmap.keys():
            raise ValueError(f"invalid feature to predict {feature_to_predict}")
        
    def split(self):
        # build x
        xtr = []
        xts = []
        for f in self.input_features:
            i,j = self.cmmap[f]

            xtr.append(self.coherence_matrix_train[:,:,i,j][self.split_mask==TRAIN].real.flatten())
            xts.append(self.coherence_matrix_test[:,:,i,j][self.split_mask==TEST].real.flatten())

            if not f in ['Shh', 'Shv', 'Svv']:
                xtr.append(self.coherence_matrix_train[:,:,i,j][self.split_mask==TRAIN].imag.flatten())
                xts.append(self.coherence_matrix_test[:,:,i,j][self.split_mask==TEST].imag.flatten())

        self.xtr = np.r_[xtr].T
        self.xts = np.r_[xts].T

        # build y
        i,j = self.cmmap[self.feature_to_predict]
        self.ytr = self.coherence_matrix_train[:,:,i,j][self.split_mask==TRAIN].real
        self.yts = self.coherence_matrix_test[:,:,i,j][self.split_mask==TEST].real
        
        if self.fit_log:
            self.ytr = np.log(self.ytr)
            self.yts = np.log(self.yts)

        return self
    
    def plot_distributions(self):
        n = self.xtr.shape[1]+1
        for ax,i in subplots(n, usizex=4):
            
            if i < n-1:
                xtri = self.xtr[:,i]
                a,b = np.percentile(xtri, [1,99])
                xtri = xtri[(xtri>a)&(xtri<b)]
                
                xtsi = self.xts[:,i]
                a,b = np.percentile(xtsi, [1,99])
                xtsi = xtsi[(xtsi>a)&(xtsi<b)]
                
                plt.hist(xtri, bins=100, alpha=.5, density=True, label='train')
                plt.hist(xtsi, bins=100, alpha=.5, density=True, label='test')
                plt.grid()
                plt.title(f"distribution of input column {i}")
                plt.legend();                
            else:
                a,b = np.percentile(self.ytr, [1,99])
                ytri = self.ytr[(self.ytr>a)&(self.ytr<b)]
                
                a,b = np.percentile(self.yts, [1,99])
                ytsi = self.yts[(self.yts>a)&(self.yts<b)]

                plt.hist(ytri, bins=100, alpha=.5, density=True, label='train')
                plt.hist(ytsi, bins=100, alpha=.5, density=True, label='test')
                plt.grid()
                plt.title(f"distribution of predictive target {self.feature_to_predict}")
                plt.legend();
                
    def set_estimator(self, estimator):
        self.estimator = estimator
        return self
    
    def fit(self):
        self.estimator.fit(self.xtr, self.ytr)
        
        self.predstr = self.estimator.predict(self.xtr)
        self.predsts = self.estimator.predict(self.xts)

        # we use mean absolute error
        self.errtr = np.mean(np.abs(self.predstr - self.ytr))
        self.errts = np.mean(np.abs(self.predsts - self.yts))
        return self

        
    def plot_predictions(self):
        for ax,i in subplots(2, usizex=5, usizey=5):
            if i==0: 
                plt.scatter(self.ytr, self.predstr, alpha=.1, s=10); 
                plt.title(f"train mae {self.errtr:.3f}") 
                plt.xlim(*np.percentile(self.ytr, [0,99]))
                plt.ylim(*np.percentile(self.predstr, [0,99]))
            if i==1: 
                plt.scatter(self.yts, self.predsts, alpha=.1, s=10); 
                plt.title(f"test mae {self.errts:.3f}")
                plt.xlim(*np.percentile(self.yts, [0,99]))
                plt.ylim(*np.percentile(self.predsts, [0,99]))
            plt.grid()


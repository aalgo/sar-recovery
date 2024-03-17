import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt

TRAIN, TEST, VAL = 0,1,2

class Trainer:
    
    def __init__(self, scatter_matrix_train, scatter_matrix_test, 
                       norm_coherence_train, norm_coherence_test, 
                       split_mask,
                       input_features, feature_to_predict):
        
        self.scatter_matrix_train = scatter_matrix_train
        self.scatter_matrix_test = scatter_matrix_test
        self.norm_coherence_train = norm_coherence_train
        self.norm_coherence_test = norm_coherence_test
        self.split_mask = split_mask
        self.input_features = input_features
        self.feature_to_predict = feature_to_predict

        self.scmap = {'HH': (0,0), 'VV': (1,1), 'HV': (0,1), 'VH': (1,0) }
        self.ncmap = {'d1':0, 'd2':1, 'd3':2,  'rho12.real':3, 'rho12.imag':4, 'rho13.real':5, 'rho13.imag':6, 'rho23.real':7, 'rho23.imag':8}

        if not (scatter_matrix_train.shape[:2] == scatter_matrix_test.shape[:2] == norm_coherence_train.shape[:2] == norm_coherence_test.shape[:2]):
            raise ValueError("all scatter matrices and normed coherences must have the same pixel size")

        if not (scatter_matrix_train.shape[2:] == scatter_matrix_test.shape[2:] == (2,2)):
            raise ValueError("scatter matrices must be of shape [h,w,2,2]")

        if not (norm_coherence_train.shape[2:] == norm_coherence_test.shape[2:] == (9,)):
            raise ValueError("normed coeherence must be of shape [h,w,9]")

        if sum([i in self.scmap.keys() for i in input_features])!=len(input_features):
            raise ValueError(f"invalid input features {input_features}")

        if not feature_to_predict in self.ncmap.keys():
            raise ValueError(f"invalid feature to predict {feature_to_predict}")
        
    def split(self):
        # build x
        xtr = []
        xts = []
        for f in self.input_features:
            i,j = self.scmap[f]

            xtr.append(self.scatter_matrix_train[:,:,i,j][self.split_mask==TRAIN].real.flatten())
            xtr.append(self.scatter_matrix_train[:,:,i,j][self.split_mask==TRAIN].imag.flatten())
            xts.append(self.scatter_matrix_test[:,:,i,j][self.split_mask==TEST].real.flatten())
            xts.append(self.scatter_matrix_test[:,:,i,j][self.split_mask==TEST].imag.flatten())

        self.xtr = np.r_[xtr].T
        self.xts = np.r_[xts].T

        # build y
        i = self.ncmap[self.feature_to_predict]
        self.ytr = self.norm_coherence_train[:,:,i][self.split_mask==TRAIN]
        self.yts = self.norm_coherence_test[:,:,i][self.split_mask==TEST]
        
        return self
    
    def plot_distributions(self):
        n = len(self.input_features)+1
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
                plt.title(f"distribution of input {self.input_features[i]}")
                plt.legend();                
            else:
                plt.hist(self.ytr, bins=100, alpha=.5, density=True, label='train')
                plt.hist(self.yts, bins=100, alpha=.5, density=True, label='test')
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
            if i==0: plt.scatter(self.ytr, self.predstr, alpha=.1, s=10); plt.title(f"train mae {self.errtr:.3f}") 
            if i==1: plt.scatter(self.yts, self.predsts, alpha=.1, s=10); plt.title(f"test mae {self.errts:.3f}")
            plt.grid()
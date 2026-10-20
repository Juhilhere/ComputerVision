import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_val_predict,GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
for n in ['uniform','early','tail']:
 a=joblib.load('mvit_output/train_mvit_'+n+'.joblib')['X']; b=joblib.load('mvit_output/test_mvit_'+n+'.joblib')['X']; X=np.r_[a,b]; d=np.r_[np.zeros(len(a)),np.ones(len(b))]
 m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
 pr=cross_val_predict(m,X,d,cv=StratifiedKFold(5,shuffle=True,random_state=7),method='predict_proba')[:,1]
 print(n,'domain',roc_auc_score(d,pr),'mean norm',np.linalg.norm(a,axis=1).mean(),np.linalg.norm(b,axis=1).mean())

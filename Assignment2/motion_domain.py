import joblib,numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
p='motion_output2/'
a=joblib.load(p+'train_motion_events.joblib')['X']; b=joblib.load(p+'test_motion_events.joblib')['X']
X=np.vstack([a,b]); y=np.r_[np.zeros(len(a)),np.ones(len(b))]
m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
pr=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=7),method='predict_proba')[:,1]
print('motion domain',roc_auc_score(y,pr),a.mean(0)[:5],b.mean(0)[:5])

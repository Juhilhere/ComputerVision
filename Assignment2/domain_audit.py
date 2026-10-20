import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
p='kaggle_output/'
for n in ['r3d_features','style']:
 a=joblib.load(p+'train_'+n+'.joblib')['X']; b=joblib.load(p+'test_'+n+'.joblib')['X']
 X=np.vstack([a,b]); y=np.r_[np.zeros(len(a)),np.ones(len(b))]
 m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
 pred=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=7),method='predict_proba',n_jobs=1)[:,1]
 print(n,'domain AUC',roc_auc_score(y,pred),'train/test norm',np.linalg.norm(a,axis=1).mean(),np.linalg.norm(b,axis=1).mean())

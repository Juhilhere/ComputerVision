import joblib,pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold,StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier,RandomForestClassifier
M=joblib.load('motion_output2/train_motion_events.joblib')['X']; T=joblib.load('motion_output2/test_motion_events.joblib')['X']; df=pd.read_csv('data_meta/train_labels.csv'); y=df.label.values; g=df.group_id.values
# domain ratio using motion only and motion + style
S=joblib.load('kaggle_output/train_style.joblib')['X']; ST=joblib.load('kaggle_output/test_style.joblib')['X']
for name,X,Z in [('motion',M,T),('motionstyle',np.c_[M,S],np.c_[T,ST])]:
 d=np.r_[np.zeros(len(X)),np.ones(len(Z))]; XX=np.r_[X,Z]
 dom=make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=1000)); dom.fit(XX,d); q=dom.predict_proba(X)[:,1]
 # density p(target)/p(source), priors correction
 w=np.clip(q/(1-q+1e-4),.2,5); w/=w.mean()
 print(name,'weights',np.percentile(w,[0,10,50,90,99,100]))
 for typ in ['hgb','et']:
  o=np.zeros(len(y))
  for fold,(a,b) in enumerate(GroupKFold(5).split(X,y,g)):
   if typ=='hgb': m=HistGradientBoostingClassifier(max_iter=260,learning_rate=.05,max_leaf_nodes=15,l2_regularization=4,random_state=7+fold)
   else: m=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=5,max_features=.8,n_jobs=-1,random_state=7+fold)
   m.fit(X[a],y[a],sample_weight=w[a]); o[b]=m.predict_proba(X[b])[:,1]
  print(typ,roc_auc_score(y,o))

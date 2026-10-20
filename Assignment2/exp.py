import joblib, pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from scipy.linalg import fractional_matrix_power
tr=joblib.load('kaggle_output/train_r3d_features.joblib')['X']; te=joblib.load('kaggle_output/test_r3d_features.joblib')['X']
s=joblib.load('kaggle_output/train_style.joblib')['X']; st=joblib.load('kaggle_output/test_style.joblib')['X']
y=pd.read_csv('data_meta/train_labels.csv').label.to_numpy(); g=pd.read_csv('data_meta/train_labels.csv').group_id.to_numpy()
print(tr.shape,te.shape,s.shape, y.mean())

def evalx(X, XT, name, model='lr'):
  oo=np.zeros(len(y)); cv=GroupKFold(5)
  for i,(a,b) in enumerate(cv.split(X,y,g)):
    if model=='lr': m=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=400))
    elif model=='hgb': m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=31,l2_regularization=2,random_state=i)
    elif model=='et': m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.7,n_jobs=-1,random_state=i,class_weight='balanced')
    m.fit(X[a],y[a]); oo[b]=m.predict_proba(X[b])[:,1]
  print(name,roc_auc_score(y,oo)); return oo
for X,XT,name in [(tr,te,'deep'),(s,st,'style'),(np.c_[tr,s],np.c_[te,st],'concat')]:
  evalx(X,XT,name,'lr')
# target featurewise rank transform combined
for kind in ['rank','allstd','coral']:
  if kind=='rank':
    from scipy.stats import rankdata
    allx=np.vstack([tr,te]); ranks=rankdata(allx,axis=0,method='average')/(len(allx)+1)
    X=ranks[:len(tr)]; XT=ranks[len(tr):]
  elif kind=='allstd':
    mu=np.vstack([tr,te]).mean(0); sd=np.vstack([tr,te]).std(0)+1e-5; X=(tr-mu)/sd; XT=(te-mu)/sd
  else:
    mu1=tr.mean(0); mu2=te.mean(0); C1=np.cov(tr-mu1,rowvar=False)+np.eye(tr.shape[1])*1e-2; C2=np.cov(te-mu2,rowvar=False)+np.eye(te.shape[1])*1e-2
    # coral A = C1^-1/2 C2^1/2
    A=fractional_matrix_power(C1,-.5)@fractional_matrix_power(C2,.5)
    X=(tr-mu1)@A+mu2; XT=(te-mu2)@A+mu2
  evalx(X,XT,'deep_'+kind,'lr')
# rank/standardize style and concat
for X,XT,name in [(np.c_[tr,s],np.c_[te,st],'concat_allstd'),(np.c_[np.vstack([tr,te]),np.vstack([s,st])][:len(tr)],np.vstack([tr,te])[len(tr):],'dummy')]: pass

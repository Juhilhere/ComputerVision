import joblib,pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
tr=joblib.load('kaggle_output/train_r3d_features.joblib')['X']; te=joblib.load('kaggle_output/test_r3d_features.joblib')['X']; s=joblib.load('kaggle_output/train_style.joblib')['X']; st=joblib.load('kaggle_output/test_style.joblib')['X']; df=pd.read_csv('data_meta/train_labels.csv'); y=df.label.values; g=df.group_id.values
# identify positive and negative rows per group

def pair_score(X,XT,name):
 o=np.zeros(len(y)); cv=GroupKFold(5)
 for fold,(a,b) in enumerate(cv.split(X,y,g)):
  sc=StandardScaler().fit(X[a]); z=sc.transform(X[a]); zb=sc.transform(X[b]); zt=sc.transform(XT)
  dif=[]
  for gg in np.unique(g[a]):
   ix=np.where(g[a]==gg)[0]; ip=ix[y[a][ix]==1]; inn=ix[y[a][ix]==0]
   if len(ip)==1 and len(inn)==1: dif.append(z[ip[0]]-z[inn[0]])
  w=np.mean(dif,axis=0); w/=np.linalg.norm(w)+1e-8
  o[b]=zb@w
 print(name,roc_auc_score(y,o))
 return o
for X,XT,n in [(tr,te,'deep'),(np.c_[tr,s],np.c_[te,st],'concat')]: pair_score(X,XT,n)
# correlation with supervised OOF logistic

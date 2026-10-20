import joblib,pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
M=joblib.load('motion_output2/train_motion_events.joblib')['X']; y=pd.read_csv('data_meta/train_labels.csv').label.values; g=pd.read_csv('data_meta/train_labels.csv').group_id.values
for pars in [dict(n_estimators=300,num_leaves=15,learning_rate=.03,min_child_samples=30,reg_lambda=5,verbosity=-1),dict(n_estimators=400,num_leaves=7,learning_rate=.03,min_child_samples=20,reg_lambda=5,verbosity=-1),dict(n_estimators=300,num_leaves=31,learning_rate=.02,min_child_samples=40,reg_lambda=10,verbosity=-1)]:
 o=np.zeros(len(y))
 for k,(a,b) in enumerate(GroupKFold(5).split(M,y,g)):
  m=LGBMClassifier(random_state=2026+k,**pars)
  m.fit(M[a],y[a],eval_set=[(M[b],y[b])],callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
  o[b]=m.predict_proba(M[b])[:,1]
 print(pars,roc_auc_score(y,o),flush=True)

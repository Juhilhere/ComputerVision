import json
from pathlib import Path
ROOT=Path(__file__).parent
CODE=r'''# Cooked or Not: stronger general-purpose Kinetics video backbone + temporal views.
# Compliance: only the competition data and general-purpose Kinetics-400 weights.
import os, glob, warnings, joblib, cv2, numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
warnings.filterwarnings('ignore'); cv2.setNumThreads(0)
SEED=2026; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; print('device',DEVICE,flush=True)

def locate_data():
    for root,dirs,files in os.walk('/kaggle/input'):
        if 'train_labels.csv' in files and 'sample_submission.csv' in files: return root
    raise FileNotFoundError('competition files')
COMP=locate_data(); TRAIN=os.path.join(COMP,'train'); TEST=os.path.join(COMP,'test')
labels=pd.read_csv(os.path.join(COMP,'train_labels.csv')); sample=pd.read_csv(os.path.join(COMP,'sample_submission.csv'))
train_ids=labels.clip_id.astype(str).tolist(); test_ids=sample.clip_id.astype(str).tolist(); y=labels.label.astype(int).to_numpy(); groups=labels.group_id.to_numpy()
print('data',COMP,labels.shape,sample.shape,flush=True)

def indices(mode):
    if mode=='uniform': return np.linspace(0,29,16).round().astype(int)
    if mode=='early': return np.arange(0,16)
    if mode=='tail': return np.arange(14,30)
    raise ValueError(mode)

def clip_tensor(folder, mode):
    files=sorted(glob.glob(os.path.join(folder,'frame_*.jpg')))
    if len(files)!=30: raise ValueError((folder,len(files)))
    frames=[]
    for i in indices(mode):
        im=cv2.imread(files[int(i)],cv2.IMREAD_COLOR)
        if im is None: raise ValueError(files[int(i)])
        im=cv2.cvtColor(cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        frames.append(im)
    a=np.asarray(frames,np.float32)/255.0 # T,H,W,C
    a=(a-np.array([0.43216,0.394666,0.37645],np.float32))/(np.array([0.22803,0.22145,0.21699],np.float32)+1e-8)
    return torch.from_numpy(a).permute(3,0,1,2) # C,T,H,W

def extract(ids,root,mode,name):
    path='/kaggle/working/'+name
    if os.path.exists(path):
        z=joblib.load(path)
        if z.get('ids')==ids and z.get('mode')==mode: return z['X']
    weights=MViT_V2_S_Weights.DEFAULT
    model=mvit_v2_s(weights=weights).to(DEVICE).eval()
    model.head=torch.nn.Identity()
    bs=4 if DEVICE=='cuda' else 1; out=[]
    with torch.inference_mode():
        for start in range(0,len(ids),bs):
            batch=torch.stack([clip_tensor(os.path.join(root,cid),mode) for cid in ids[start:start+bs]]).to(DEVICE)
            f=model(batch).float().cpu().numpy(); out.append(f)
            if start+len(batch)==len(ids) or (start//bs)%25==0: print(name,start+len(batch),'/',len(ids),flush=True)
    X=np.concatenate(out).astype('float32'); joblib.dump({'ids':ids,'mode':mode,'X':X},path); del model
    if DEVICE=='cuda': torch.cuda.empty_cache()
    return X

# Three views cover a crash at either end of the 30-frame clip.
V={}
for mode in ['uniform','early','tail']:
    V[mode]=(extract(train_ids,TRAIN,mode,'train_mvit_'+mode+'.joblib'),extract(test_ids,TEST,mode,'test_mvit_'+mode+'.joblib'))
print('MViT matrices',[(k,a.shape,b.shape) for k,(a,b) in V.items()],flush=True)

def get_old(name,ids):
    for root,dirs,files in os.walk('/kaggle/input'):
        if name in files:
            z=joblib.load(os.path.join(root,name))
            if z.get('ids')==ids: return z['X']
    raise FileNotFoundError(name)
Rtr=get_old('train_r3d_features.joblib',train_ids); Rte=get_old('test_r3d_features.joblib',test_ids)
Mtr=get_old('train_motion_events.joblib',train_ids); Mte=get_old('test_motion_events.joblib',test_ids)
print('reused',Rtr.shape,Mtr.shape,flush=True)

def cv_lr(X,C=.10):
    o=np.zeros(len(y),np.float32)
    for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000))
        m.fit(X[tr],y[tr]); o[va]=m.predict_proba(X[va])[:,1]
    return o

def cv_hgb(X):
    o=np.zeros(len(y),np.float32)
    for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
        m=HistGradientBoostingClassifier(max_iter=260,learning_rate=.05,max_leaf_nodes=15,l2_regularization=4,random_state=SEED+fold)
        m.fit(X[tr],y[tr]); o[va]=m.predict_proba(X[va])[:,1]
    return o

# Validate every independent source under the required group split.
pr=cv_lr(Rtr,.15); pm=cv_hgb(Mtr)
pviews={k:cv_lr(a,.08) for k,(a,b) in V.items()}
for n,p in [('R3D',pr),('motion HGB',pm)]+[('MViT '+k,pviews[k]) for k in pviews]: print(n,'group OOF',round(roc_auc_score(y,p),5),flush=True)
# Temporal averaging is selected from validation, then held fixed for the test prediction.
view_names=list(pviews); best_view=(0,None)
for wu in np.arange(0,1.01,.1):
  for we in np.arange(0,1.01-wu,.1):
    wt=1-wu-we; p=wu*pviews['uniform']+we*pviews['early']+wt*pviews['tail']; s=roc_auc_score(y,p)
    if s>best_view[0]: best_view=(s,(wu,we,wt))
print('best MViT view blend',best_view,flush=True)
pmvit=best_view[1][0]*pviews['uniform']+best_view[1][1]*pviews['early']+best_view[1][2]*pviews['tail']
# Search a coarse convex blend on group OOF; no public labels are used.
best=(0,None)
for wm in np.arange(0,1.01,.05):
  for wv in np.arange(0,1.01-wm,.05):
    wr=1-wm-wv; p=wm*pm+wv*pmvit+wr*pr; s=roc_auc_score(y,p)
    if s>best[0]: best=(s,(wm,wv,wr))
print('best final blend OOF',best,flush=True)

def fit_lr(X,XT,C=.10):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000)); m.fit(X,y); return m.predict_proba(XT)[:,1]
def fit_hgb(X,XT):
    m=HistGradientBoostingClassifier(max_iter=260,learning_rate=.05,max_leaf_nodes=15,l2_regularization=4,random_state=SEED); m.fit(X,y); return m.predict_proba(XT)[:,1]
ptr=fit_lr(Rtr,Rte,.15); pmt=fit_hgb(Mtr,Mte)
pvt={k:fit_lr(a,b,.08) for k,(a,b) in V.items()}
wu,we,wt=best_view[1]; pvt_mvit=wu*pvt['uniform']+we*pvt['early']+wt*pvt['tail']
wm,wv,wr=best[1]; pred=wm*pmt+wv*pvt_mvit+wr*ptr; pred=np.clip(pred,.001,.999)
out=sample[['clip_id']].copy(); out['label']=pred
assert out.clip_id.is_unique and len(out)==len(sample) and out.label.notna().all() and out.label.between(0,1).all()
out.to_csv('/kaggle/working/submission_mvit.csv',index=False)
print('READY submission_mvit.csv',out.shape,float(pred.min()),float(pred.max()),flush=True)
'''
nb={'cells':[{'cell_type':'code','metadata':{},'source':CODE,'outputs':[],'execution_count':None}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':4}
out=ROOT/'kaggle_mvit'/'cooked-or-not-mvit-temporal-ensemble.ipynb'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb),encoding='utf-8')
metadata={'id':'juhilmodi/cooked-or-not-mvit-temporal-ensemble','title':'Cooked or Not - MViT temporal ensemble','code_file':out.name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'keywords':[],'dataset_sources':[],'kernel_sources':['juhilmodi/cooked-or-not-r3d-domain-shift-experiment','juhilmodi/cooked-or-not-motion-event-transfer-experiment'],'competition_sources':['cooked-or-not'],'model_sources':[]}
(out.parent/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(out)

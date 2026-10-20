import json
from pathlib import Path

ROOT = Path(__file__).parent
CODE = r'''# Cooked or Not: motion-event experiment (competition data only).
# 30-frame physical-change signatures complement the earlier Kinetics R3D features.
import os, glob, warnings
from concurrent.futures import ThreadPoolExecutor
import cv2, joblib, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
SEED=2026
np.random.seed(SEED)

def locate_data():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'train_labels.csv' in files and 'sample_submission.csv' in files:
            return root
    raise FileNotFoundError('competition files')

COMP=locate_data(); TRAIN=os.path.join(COMP,'train'); TEST=os.path.join(COMP,'test')
labels=pd.read_csv(os.path.join(COMP,'train_labels.csv'))
sample=pd.read_csv(os.path.join(COMP,'sample_submission.csv'))
train_ids=labels.clip_id.astype(str).tolist(); test_ids=sample.clip_id.astype(str).tolist()
y=labels.label.astype(int).to_numpy(); groups=labels.group_id.to_numpy()
print('data',COMP,labels.shape,sample.shape,flush=True)

def read_frames(folder):
    files=sorted(glob.glob(os.path.join(folder,'frame_*.jpg')))
    if len(files)!=30: raise ValueError((folder,len(files)))
    frames=[]
    for p in files:
        im=cv2.imread(p,cv2.IMREAD_COLOR)
        if im is None: raise ValueError(p)
        frames.append(cv2.resize(im,(96,56),interpolation=cv2.INTER_AREA))
    return frames

def summarize(a):
    a=np.asarray(a,dtype=np.float32)
    return np.array([a.mean(),a.std(),a.max(),np.percentile(a,90),
      a[:14].mean(),a[14:].mean(),a[14:].mean()-a[:14].mean(),
      a[:14].max(),a[14:].max(),a[14:].max()-a[:14].max(),
      np.argmax(a)/max(1,len(a)-1)],dtype=np.float32)

def clip_features(folder):
    frames=read_frames(folder)
    gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
    channels=[[] for _ in range(11)]
    # Frame-to-frame signals emphasize abrupt impact and camera shake.
    for t in range(1,30):
        a=gray[t-1]; b=gray[t]
        d=cv2.absdiff(a,b).astype(np.float32)/255
        central=d[12:48,16:80]
        flow=cv2.calcOpticalFlowFarneback(a,b,None,0.5,2,11,2,5,1.1,0)
        fx,fy=flow[:,:,0],flow[:,:,1]
        mag=np.sqrt(fx*fx+fy*fy)
        medx,medy=np.median(fx),np.median(fy)
        residual=np.sqrt((fx-medx)**2+(fy-medy)**2)
        ea=cv2.Canny(a,60,140); eb=cv2.Canny(b,60,140)
        vals=[d.mean(),np.percentile(d,90),central.mean(),(d>.15).mean(),
              mag.mean(),np.percentile(mag,90),residual.mean(),
              np.percentile(residual,90),np.hypot(medx,medy),
              np.abs(ea.astype(np.float32)-eb).mean()/255,
              abs(float(a.mean())-float(b.mean()))/255]
        for k,v in enumerate(vals): channels[k].append(v)
    # Temporal event scores; normalize by within-clip baseline to resist
    # camera, weather and compression changes.
    out=[]
    for v in channels:
        z=summarize(v); out.extend(z)
        ar=np.asarray(v); base=np.median(ar[:14])+1e-3
        out.extend([ar[14:].max()/base, np.percentile(ar[14:],90)/base,
                    (ar[14:].max()-np.median(ar))/(np.std(ar[:14])+1e-3)])
    # Smooth global image statistics are used only as changes, not appearance.
    for metric in (lambda im: cv2.Laplacian(im,cv2.CV_32F).var(),
                   lambda im: im.mean(),lambda im: im.std()):
        vals=np.array([metric(g) for g in gray],np.float32)
        out.extend([vals[-10:].mean()-vals[:10].mean(),
                    vals[-10:].std()-vals[:10].std(),
                    np.max(np.abs(np.diff(vals)))/(np.std(vals[:10])+1e-3)])
    x=np.asarray(out,np.float32)
    return np.nan_to_num(x,nan=0,posinf=0,neginf=0)

def build(ids,root,name):
    cache='/kaggle/working/'+name
    if os.path.exists(cache):
        obj=joblib.load(cache)
        if obj['ids']==ids: return obj['X']
    X=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,x in enumerate(pool.map(lambda cid: clip_features(os.path.join(root,cid)),ids)):
            X.append(x)
            if (i+1)%250==0 or i+1==len(ids): print(name,i+1,'/',len(ids),flush=True)
    X=np.stack(X)
    joblib.dump({'ids':ids,'X':X},cache)
    return X

Mtr=build(train_ids,TRAIN,'train_motion_events.joblib')
Mte=build(test_ids,TEST,'test_motion_events.joblib')
print('motion matrices',Mtr.shape,Mte.shape,flush=True)

def get_old(name,ids):
    for root,dirs,files in os.walk('/kaggle/input'):
        if name in files:
            obj=joblib.load(os.path.join(root,name))
            if obj['ids']==ids: return obj['X']
    raise FileNotFoundError(name)

Rtr=get_old('train_r3d_features.joblib',train_ids)
Rte=get_old('test_r3d_features.joblib',test_ids)
Str=get_old('train_style.joblib',train_ids)
Ste=get_old('test_style.joblib',test_ids)
print('old matrices',Rtr.shape,Str.shape,flush=True)

def cv_predict(X,model_name):
    oof=np.zeros(len(y),np.float32)
    for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups)):
        if model_name=='lr':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=500))
        elif model_name=='hgb':
            model=HistGradientBoostingClassifier(max_iter=220,learning_rate=.05,
                     max_leaf_nodes=15,l2_regularization=4,random_state=SEED+fold)
        else:
            model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,
                     max_features=.8,n_jobs=-1,random_state=SEED+fold)
        model.fit(X[tr],y[tr]); oof[va]=model.predict_proba(X[va])[:,1]
    return oof

# Within-group CV is still optimistic about test domain; report each branch.
base=cv_predict(Rtr,'lr')
style=cv_predict(Str,'hgb')
motion_lr=cv_predict(Mtr,'lr')
motion_hgb=cv_predict(Mtr,'hgb')
motion_et=cv_predict(Mtr,'et')
motion=.35*motion_lr+.35*motion_hgb+.30*motion_et
for name,p in [('R3D',base),('style',style),('motion LR',motion_lr),
               ('motion HGB',motion_hgb),('motion ET',motion_et),('motion ensemble',motion)]:
    print(name,'group OOF',round(roc_auc_score(y,p),5),flush=True)
for w in [0,.25,.5,.75,1.0]:
    p=(1-w)*base+w*motion
    print('blend motion',w,'group OOF',round(roc_auc_score(y,p),5),flush=True)

# The candidate gives event evidence the majority weight because the
# training/test negative definitions differ. This weight is fixed in advance.
def full_predict(X,XT,model_name):
    if model_name=='lr': model=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=500))
    elif model_name=='hgb': model=HistGradientBoostingClassifier(max_iter=220,learning_rate=.05,
             max_leaf_nodes=15,l2_regularization=4,random_state=SEED)
    else: model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,
             max_features=.8,n_jobs=-1,random_state=SEED)
    model.fit(X,y)
    return model.predict_proba(XT)[:,1]

pbase=full_predict(Rtr,Rte,'lr')
pmotion=.35*full_predict(Mtr,Mte,'lr')+.35*full_predict(Mtr,Mte,'hgb')+.30*full_predict(Mtr,Mte,'et')
pred=.40*pbase+.60*pmotion
pred=np.clip(pred,.001,.999)
out=sample[['clip_id']].copy(); out['label']=pred
assert out.clip_id.is_unique and len(out)==1750 and out.label.notna().all() and out.label.between(0,1).all()
out.to_csv('/kaggle/working/submission_motion.csv',index=False)
print('READY submission_motion.csv',out.shape,float(pred.min()),float(pred.max()),flush=True)
'''

nb = {
    'cells': [{'cell_type': 'code', 'metadata': {}, 'source': CODE, 'outputs': [], 'execution_count': None}],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
out = ROOT / 'kaggle_motion' / 'cooked-or-not-motion-events.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb), encoding='utf-8')
metadata = {
    'id': 'juhilmodi/cooked-or-not-motion-events',
    'title': 'Cooked or Not - motion event transfer experiment',
    'code_file': out.name,
    'language': 'python', 'kernel_type': 'notebook',
    'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
    'enable_internet': False, 'keywords': [], 'dataset_sources': [],
    'kernel_sources': ['juhilmodi/cooked-or-not-r3d-domain-shift-experiment'],
    'competition_sources': ['cooked-or-not'], 'model_sources': [],
}
(out.parent / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(out)

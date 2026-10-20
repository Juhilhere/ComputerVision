import torchvision
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, swin3d_t, Swin3D_T_Weights
m=mvit_v2_s(weights=None)
print(m)
print('head',m.head)
s=swin3d_t(weights=None)
print('swin head',s.head)

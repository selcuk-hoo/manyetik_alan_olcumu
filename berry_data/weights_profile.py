# w_i agirlik profili: ogrenilen cekirdek halka boyunca nerede yogun? (berry.md §5)
# tracker GEREKMEZ; run1/run2 npz'den calisir. Cikti: berry_weights.png
import numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_D=os.path.dirname(os.path.abspath(__file__))
d1=np.load(os.path.join(_D,"run1_data.npz")); d2=np.load(os.path.join(_D,"run2_data.npz"))
f=np.concatenate([d1['f'],d2['f']]); xo=np.vstack([d1['xo'],d2['xo']]); yo=np.vstack([d1['yo'],d2['yo']])
M,L=xo.shape
def fit_w(N,lam):
    idx=np.linspace(0,L-1,N).astype(int); X=xo[:,idx]*yo[:,idx]
    mu=X.mean(0); sd=X.std(0)+1e-30; Xs=(X-mu)/sd
    w=np.linalg.solve(Xs.T@Xs+lam*np.eye(N),Xs.T@(f-f.mean()))
    return idx/L, w/np.abs(w).max()
plt.figure(figsize=(7,4))
for N,lam,c in [(24,10,'C0'),(48,10,'C1'),(24,30,'C2'),(48,30,'C3')]:
    s,w=fit_w(N,lam); plt.plot(s,w,'o-',ms=3,color=c,alpha=0.7,label='N=%d λ=%d'%(N,lam))
plt.plot(np.arange(L)/L, np.sqrt((xo**2).mean(0))/np.sqrt((xo**2).mean()).max()*0.5,'k--',lw=0.8,alpha=0.4,label='|x| profil (ölçek)')
plt.axhline(0,color='gray',lw=0.5); plt.xlabel('halka konumu (tur kesri s/C)'); plt.ylabel('normalize ağırlık w_i')
plt.title('Sahte-EDM kuplaj ağırlığı w_i — nerede yoğun?'); plt.legend(fontsize=7); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(_D,'berry_weights.png'),dpi=140)
print('Kaydedildi: berry_weights.png')

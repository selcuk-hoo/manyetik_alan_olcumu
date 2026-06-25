# Bilineer çekirdek fit: f ~ Σ W_ij x_i y_j, lokallik kısıtlı, permütasyon-testli.
# Modeller: (1) uniform <xy>  (2) ağırlıklı diagonal Σ w_i x_i y_i
#           (3) + Berry-alan komşu Σ a_i (x_i y_{i+1} - x_{i+1} y_i)
# Amaç: ML bu fonksiyoneli (sahte-R² olmadan) bulabilir mi? Berry yapısı gerçek mi?
import numpy as np, os
_D=os.path.dirname(os.path.abspath(__file__))
d1=np.load(os.path.join(_D,"run1_data.npz")); d2=np.load(os.path.join(_D,"run2_data.npz"))
f=np.concatenate([d1["f"],d2["f"]]); xo=np.vstack([d1["xo"],d2["xo"]]); yo=np.vstack([d1["yo"],d2["yo"]])
w=np.concatenate([d1["w"],d2["w"]]); M=len(f)
print("toplam config:", M, " gruplar:", {ww:int((w==ww).sum()) for ww in sorted(set(w))})

def down(a,N):
    idx=np.linspace(0,a.shape[1]-1,N).astype(int); return a[:,idx]

def loo_r2(X,y,lam):
    n=len(y); pred=np.zeros(n); mu=X.mean(0); sd=X.std(0)+1e-30; Xs=(X-mu)/sd
    for i in range(n):
        tr=[j for j in range(n) if j!=i]; A=Xs[tr]; b=y[tr]-y[tr].mean()
        W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@b); pred[i]=Xs[i]@W+y[tr].mean()
    return 1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)

def best_loo(X,y): return max(loo_r2(X,y,l) for l in (0.3,1,3,10,30,100))

def perm_null(X,y,nperm=100):
    rng=np.random.default_rng(0); null=[]
    for _ in range(nperm): null.append(best_loo(X,rng.permutation(y)))
    return np.array(null)

for N in (12,16,24):
    X=down(xo,N); Y=down(yo,N)
    diag=X*Y                                   # x_i y_i
    berry=X[:,:-1]*Y[:,1:]-X[:,1:]*Y[:,:-1]    # x_i y_{i+1} - x_{i+1} y_i
    feat_uniform=diag.sum(1,keepdims=True)
    feat_diag=diag
    feat_full=np.hstack([diag,berry])
    print("\n=== N=%d nokta ==="%N)
    cu=np.corrcoef(feat_uniform[:,0],f)[0,1]
    print("  (1) uniform <xy>      : corr=%+.2f  R²=%.2f"%(cu,cu**2))
    for name,Xf in [("(2) agirlikli diagonal",feat_diag),("(3) +Berry komsu",feat_full)]:
        r=best_loo(Xf,f); nul=perm_null(Xf,f)
        flag="GERCEK" if r>np.percentile(nul,95) else "SUPHELI(asiri-uyum)"
        print("  %-22s: LOO-R²=%.2f  null(ort/95%%/maks)=%.2f/%.2f/%.2f  -> %s"%(
            name,r,nul.mean(),np.percentile(nul,95),nul.max(),flag))
# en iyi N icin diagonal agirliklarini goster (yorumlanabilirlik)
N=16; X=down(xo,N); Y=down(yo,N); diag=X*Y
mu=diag.mean(0); sd=diag.std(0)+1e-30; Xs=(diag-mu)/sd
Wd=np.linalg.solve(Xs.T@Xs+10*np.eye(N),Xs.T@(f-f.mean()))
print("\nOgrenilen diagonal agirliklar w_i (N=16, kuplaj nerede yogun?):")
print("  ", " ".join("%+.2f"%v for v in Wd/np.abs(Wd).max()))

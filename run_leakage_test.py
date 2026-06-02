#!/usr/bin/env python3
"""Build bitince test_kmod + reconstruction'ı iki bazla çalıştırır.

1) Bekle: R_dy_2.npy hazır olsun (build tamamlandı).
2) test_kmod_reconstruction.py → Δy üret.
3) reconstruction.py → baz = truth (k=2,4,6,8).
4) recon_k_list_dy=[2] ekle → reconstruction.py (sızıntı testi).
5) params.json'u orijinaline geri al.
"""
import json, os, time, subprocess, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# 1) build'i bekle
print("[driver] R_dy_2.npy bekleniyor...", flush=True)
while not os.path.exists("R_dy_2.npy"):
    time.sleep(20)
time.sleep(5)  # yazma tamamlansın
print("[driver] build hazir. test_kmod calisiyor...", flush=True)

# 2) Δy üret
subprocess.run(["python3", "test_kmod_reconstruction.py"], check=True)

# 3) varsayılan baz (truth = k=2,4,6,8)
print("\n[driver] ===== RECON #1: baz = truth (k=2,4,6,8) =====", flush=True)
subprocess.run(["python3", "reconstruction.py"], check=True)

# 4) sızıntı testi: baz yalnız k=2
shutil.copy("params.json", "params.json.bak")
with open("params.json") as f:
    cfg = json.load(f)
cfg["recon_k_list_dy"] = [2]
with open("params.json", "w") as f:
    json.dump(cfg, f, indent=4)
print("\n[driver] ===== RECON #2: baz = {k=2} (sizinti testi) =====", flush=True)
subprocess.run(["python3", "reconstruction.py"], check=True)

# 5) geri al
shutil.move("params.json.bak", "params.json")
print("\n[driver] tamamlandi. params.json geri yuklendi.", flush=True)

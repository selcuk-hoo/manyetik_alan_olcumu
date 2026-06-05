#!/usr/bin/env python3
"""fig_6_combined_systematics.py — ŞEKİL 6: kombinat sistematik hata bütçesi.

Çıktı: fig_6_combined_systematics.png
Kaynak betik: test_combined_systematics.py
"""
import subprocess, shutil, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "test_combined_systematics.py"], check=True)
shutil.copy("combined_systematics.png", "fig_6_combined_systematics.png")
print("→ fig_6_combined_systematics.png")

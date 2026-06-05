#!/usr/bin/env python3
"""fig_4_reconstruction_quality.py — ŞEKİL 4: geri çatım kalitesi kıyası.

Çıktı: fig_4_reconstruction_quality.png
Kaynak betik: test_reconstruction_quality.py
"""
import subprocess, shutil, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "test_reconstruction_quality.py"], check=True)
shutil.copy("reconstruction_quality.png", "fig_4_reconstruction_quality.png")
print("→ fig_4_reconstruction_quality.png")

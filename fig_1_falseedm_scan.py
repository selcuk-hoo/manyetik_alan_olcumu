#!/usr/bin/env python3
"""fig_1_falseedm_scan.py — ŞEKİL 1: sahte-EDM oranı |dS_y/dt| vs k taraması.

Çıktı: fig_1_falseedm_scan.png
Kaynak betik: false_edm_mode_scan.py
"""
import subprocess, shutil, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "false_edm_mode_scan.py"], check=True)
shutil.copy("false_edm_mode_scan.png", "fig_1_falseedm_scan.png")
print("→ fig_1_falseedm_scan.png")

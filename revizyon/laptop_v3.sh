#!/usr/bin/env bash
# laptop_v3.sh — Üçüncü tur koşular (kusurlu polarite: 12 desen + σ_d taraması), tamamen otomatik.
# Ayrıntı ve gerekçe: LAPTOP_PROMPT_v3.md. Kesilirse aynı komutla yeniden çalıştır (kaldığı yerden devam).
#
# Kullanım (repo kökünden):   nohup bash revizyon/laptop_v3.sh 10 > laptop_v3.log 2>&1 &
#   argüman: işçi sayısı (varsayılan 10)
# İlerleme: tail -f laptop_v3.log
set -u
W="${1:-10}"
BRANCH="claude/brave-allen-92wn61"
cd "$(dirname "$0")/.."            # repo kökü
ROOT="$(pwd)"
RES="revizyon/sonuclar/kusurlu_polarite.jsonl"
log() { echo "[$(date '+%d.%m %H:%M')] $*"; }

push() {   # sonuçları commit + push (4 deneme, artan bekleyiş)
  git add -f "$RES" revizyon/sonuclar/LAPTOP_OZET_v3.md 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "test: $1 (laptop)" || return 0
  for i in 1 2 3 4; do
    git push -q origin "$BRANCH" && { log "push tamam"; return 0; }
    log "push başarısız, ${i}. deneme"; sleep $((2 ** i * 5))
    git pull -q --rebase origin "$BRANCH"
  done
  log "UYARI: push başarısız, sonuçlar yerelde commit'li"
}

# 1) güncelle (yerelde aynı adlı izlenmeyen .npy varsa yedekle)
mkdir -p ~/npy_yedek
for f in R_dx_1.npy R_dy_1.npy R_dx_cw.npy R_dy_cw.npy revizyon/sonuclar/D_*.npy revizyon/sonuclar/Rtrue_*_g20.npy; do
  [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && mv "$f" ~/npy_yedek/ && log "yedeklendi: $f"
done
git pull -q origin "$BRANCH" || { log "HATA: git pull başarısız"; exit 1; }

# 2) derle ve doğrula
bash build_integrator.sh >/dev/null || { log "HATA: build_integrator.sh"; exit 1; }
( cd revizyon && g++ -O3 -shared -fPIC -std=c++17 integrator_rev.cpp -o lib_integrator_rev.so ) \
  || { log "HATA: integrator_rev derlenemedi"; exit 1; }
( cd revizyon && python3 -c "import integrator_rev, kusurlu_polarite" ) || { log "HATA: import"; exit 1; }
log "derleme tamam; mevcut sonuç özeti (bulut, 3 desen):"
( cd revizyon && python3 kusurlu_polarite.py --ozet | tail -12 )

# 3) periyodik push (her 2 saatte bir, arka planda)
( while true; do sleep 7200; cd "$ROOT" && push "kusurlu polarite ara sonuç"; done ) &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null' EXIT

# 4) İş 1: σ_d = 1e-5, desen 3–11
log "İş 1 başlıyor (σ_d=1e-5, seed 3-11, -w $W)"
( cd revizyon && python3 kusurlu_polarite.py -w "$W" --sigd 1e-5 --seeds 3 4 5 6 7 8 9 10 11 ) \
  | grep -v "t = \|target_quad" || log "UYARI: İş 1 hata ile bitti (yeniden çalıştır)"
push "kusurlu polarite iş 1 (σ_d=1e-5, 12 desen)"

# 5) İş 2: σ_d taraması, desen 0–5 (her σ_d ayrı; biri çökerse diğerleri sürer)
for SD in 1e-6 3e-5 1e-4; do
  log "İş 2: σ_d=$SD başlıyor"
  ( cd revizyon && python3 kusurlu_polarite.py -w "$W" --sigd "$SD" --seeds 0 1 2 3 4 5 ) \
    | grep -v "t = \|target_quad" || log "UYARI: σ_d=$SD hata ile bitti (atlandı)"
  push "kusurlu polarite σ_d=$SD"
done

# 6) özet
{
  echo "# Laptop üçüncü tur — kusurlu polarite özeti"
  echo
  echo "Tarih: $(date '+%d.%m.%Y %H:%M'). Makine: $(hostname), $(nproc) çekirdek, -w $W. Betik: revizyon/laptop_v3.sh."
  echo "Ayrıntı: LAPTOP_PROMPT_v3.md. E = ½|odd₊ + odd₋|, hedef (1e-9) biriminde."
  echo
  echo '```'
  ( cd revizyon && python3 kusurlu_polarite.py --ozet )
  echo '```'
  echo
  echo "Hata/uyarı satırları (laptop_v3.log):"
  grep -i "uyarı\|hata\|error\|nan" "$ROOT/laptop_v3.log" 2>/dev/null | head -20 || true
} > revizyon/sonuclar/LAPTOP_OZET_v3.md
push "laptop v3 özeti"
log "BİTTİ. Özet: revizyon/sonuclar/LAPTOP_OZET_v3.md"

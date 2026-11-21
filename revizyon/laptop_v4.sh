#!/usr/bin/env bash
# laptop_v4.sh — Dördüncü tur: kusurlu polarite testine ORTAK alan sürüklenmesi (ortak HV beslemesi) durumları.
# Yeni durumlar: dc6_yok, dc5_yok (iki polarite koşusu arasında bütün saptırıcılara Δε_c = 1e-6, 1e-5),
# dc5_E4 (Δε_c = 1e-5, iki demetli fitin saptırıcı kestiriminin ortalamasıyla ortak kaynak yeniden ayarlanır).
# Önceki durumlar dosyada olduğu için atlanır; yalnız yeni durumlar hesaplanır (~200 görev).
# Gerekçe: makale_orbit_bastirma_tr_v2_uzun.tex, 'Ortak besleme' paragrafı. Kesilirse aynı komutla yeniden çalıştır (kaldığı yerden devam).
#
# Kullanım (repo kökünden):   nohup bash revizyon/laptop_v4.sh 10 > laptop_v4.log 2>&1 &
#   argüman: işçi sayısı (varsayılan 10)
# İlerleme: tail -f laptop_v4.log
set -u
W="${1:-10}"
BRANCH="claude/brave-allen-92wn61"
cd "$(dirname "$0")/.."            # repo kökü
ROOT="$(pwd)"
RES="revizyon/sonuclar/kusurlu_polarite.jsonl"
log() { echo "[$(date '+%d.%m %H:%M')] $*"; }

push() {   # sonuçları commit + push (4 deneme, artan bekleyiş)
  git add -f "$RES" revizyon/sonuclar/LAPTOP_OZET_v4.md 2>/dev/null
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

# 4) Tek iş: bütün σ_d'ler, önceki desen kümeleriyle (yalnız yeni durumlar hesaplanır)
for SPEC in "0 0 1 2" "1e-6 0 1 2 3 4 5" "1e-5 0 1 2 3 4 5 6 7 8 9 10 11" "3e-5 0 1 2 3 4 5" "1e-4 0 1 2 3 4 5"; do
  set -- $SPEC; SD=$1; shift
  log "σ_d=$SD, seed $* başlıyor"
  ( cd revizyon && python3 kusurlu_polarite.py -w "$W" --sigd "$SD" --seeds "$@" ) \
    | grep -v "t = \|target_quad" || log "UYARI: σ_d=$SD hata ile bitti (atlandı)"
  push "kusurlu polarite ortak sürüklenme σ_d=$SD"
done

# 6) özet
{
  echo "# Laptop dördüncü tur — kusurlu polarite, ortak alan sürüklenmesi"
  echo
  echo "Tarih: $(date '+%d.%m.%Y %H:%M'). Makine: $(hostname), $(nproc) çekirdek, -w $W. Betik: revizyon/laptop_v4.sh."
  echo "Yeni satırlar: ortak Δε 1e-6 / 1e-5 (düzeltme yok) ve 1e-5 (fitle ortak trim). E = ½|odd₊ + odd₋|, hedef (1e-9) biriminde."
  echo
  echo '```'
  ( cd revizyon && python3 kusurlu_polarite.py --ozet )
  echo '```'
  echo
  echo "Hata/uyarı satırları (laptop_v4.log):"
  grep -i "uyarı\|hata\|error\|nan" "$ROOT/laptop_v4.log" 2>/dev/null | head -20 || true
} > revizyon/sonuclar/LAPTOP_OZET_v4.md
push "laptop v4 özeti"
log "BİTTİ. Özet: revizyon/sonuclar/LAPTOP_OZET_v4.md"

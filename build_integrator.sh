#!/usr/bin/env bash
# build_integrator.sh — integrator.cpp'yi mevcut platform için derler.
#
# NEDEN: integrator.cpp güncellendiğinde (quad misalignment, quad tilt,
# k-modülasyon destekleri) derlenmiş kütüphane bayatlarsa, integrator.py
# yeni quad_dy/quad_tilt dizilerini gönderir ama eski kütüphane bunları
# UYGULAMAZ → tüm simülasyonlar misalignmentsiz davranır (sessiz hata).
# Kaynağı her değiştirdiğinde bu betiği çalıştır.
#
# Kullanım:  bash build_integrator.sh
set -e
cd "$(dirname "$0")"

FLAGS="-O3 -shared -fPIC -std=c++17"
SRC="integrator.cpp"

case "$(uname -s)" in
  Darwin)
    OUT="integrator.dylib"
    CXX="${CXX:-clang++}"
    ;;
  Linux)
    OUT="lib_integrator.so"
    CXX="${CXX:-g++}"
    ;;
  *)
    echo "Bilinmeyen platform: $(uname -s). Elle derleyin." >&2
    exit 1
    ;;
esac

echo "Derleniyor: $CXX $FLAGS $SRC -o $OUT"
"$CXX" $FLAGS "$SRC" -o "$OUT"
echo "Tamam → $OUT ($(uname -s))"
echo "Doğrulama: python3 -c \"import integrator; print('yüklendi')\""

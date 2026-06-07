#!/bin/bash
echo "Waiting for make_fig1.py to finish (if it's still running)..."

# Now run the rest sequentially
echo "Running make_fig2.py..."
python3 make_fig2.py

echo "Running make_fig3.py..."
python3 make_fig3.py

echo "Running make_fig4.py..."
python3 make_fig4.py

echo "Running make_fig5.py..."
python3 make_fig5.py

echo "Running make_fig6.py..."
python3 make_fig6.py

echo "All figures generated."
echo "Running make_fig7.py..."
python3 make_fig7.py

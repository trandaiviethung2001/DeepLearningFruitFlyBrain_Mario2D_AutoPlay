#!/bin/bash
cd "$(dirname "$0")"

clear
echo "======================================================================"
echo "🎮 DROSOPHILA MALECNS CONNECTOME MARIO 2D - LIVE GUI WINDOW"
echo "======================================================================"
echo "Opening native NES game window on your macOS Desktop at 30 FPS..."
echo "Policy: Drosophila MaleCNS Connectome-Constrained Recurrent Network"
echo "Feature: Auto-Respawn & Continuous Play (Window will stay open)"
echo "To exit: Close the game window or press Ctrl+C in this terminal"
echo "======================================================================"
echo ""

./.venv/bin/python play_game.py --model connectome --render --temperature 0.6 --jump-hold 8 --fps 30 --loop --keep-open --save-gif ""

echo ""
echo "Game session finished."
read -p "Press [Enter] to close this terminal window..."


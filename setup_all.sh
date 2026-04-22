#!/bin/bash
# setup_all.sh -- run once after cloning all 3 repos side-by-side
#
# Expected layout:
#   wrk/
#     storyAI/         (this repo -- orchestrator + story generation)
#     manga-ai-bot/    (panel image generation via Stable Diffusion)
#     bubble/          (manhwa-bubbles speech bubble library)
#
# Usage:
#   cd storyAI && bash setup_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

MANGA_BOT="$PARENT_DIR/manga-ai-bot"
BUBBLE="$PARENT_DIR/bubble"

# Verify sibling folders exist
for dir in "$MANGA_BOT" "$BUBBLE"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Expected sibling directory not found: $dir"
        echo "Make sure storyAI, manga-ai-bot, and bubble are in the same parent folder."
        exit 1
    fi
done

cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -d "menv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv menv
fi

echo "Activating virtual environment..."
source menv/bin/activate

echo "Installing storyAI dependencies..."
pip install -r requirements.txt

echo "Installing manga-ai-bot (editable)..."
pip install -e "$MANGA_BOT"

echo "Installing manhwa-bubbles (editable)..."
pip install -e "$BUBBLE"

echo ""
echo "Done! All 3 projects are linked."
echo "Activate with:  source menv/bin/activate"
echo ""
echo "Quick start:"
echo "  python main.py              # Generate story text"
echo "  python generate_manga.py    # Generate manga from chapters"
echo "  python serve.py             # View in browser"

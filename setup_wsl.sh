#!/usr/bin/env bash
# Creates this server's own virtualenv (./.venv) -- nothing is shared with the
# LivePortrait / conference environments. Run inside Linux or WSL2.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt install -y libsndfile1
python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch==2.2.2 torchaudio==2.2.2
pip install git+https://github.com/facebookresearch/seamless_communication.git
pip install -r requirements.txt
pip install silero-vad
echo "done. Download models once with:  ./run_server.sh --prefetch"

#!/bin/bash

# pass username as first argument, otherwise default to "user"
if [ ! -z "$1" ]; then
    USERNAME="$1"
else
    USERNAME="user"
fi

echo "Using username: $USERNAME"

DIRECTORY=generative_models
GIT_REPO="https://github.com/ktroyan/GVM-Memorization.git"
GIT_FOLDER="GVM-Memorization"
GIT_BRANCH="nauryz"

ssh $USERNAME@euler << ENDSSH
git clone "$GIT_REPO"
cd "$GIT_FOLDER"
git fetch --all
git checkout "$GIT_BRANCH"

cd .. 
mkdir -p "$DIRECTORY"
cd "$DIRECTORY"

module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9
pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1
pip install xformers==0.0.19
pip install diffusers accelerate transformers mediapy triton scipy ftfy spacy==3.4.4 
pip install tqdm
ENDSSH
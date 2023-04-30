USERNAME="nkoisheke"
DIRECTORY=generative_models
ssh $USERNAME@euler mkdir -p $DIRECTORY

ssh $USERNAME@euler << ENDSSH
cd "$DIRECTORY"
module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9
pip install --upgrade torch torchvision torchaudio
pip install --upgrade xformers
ENDSSH

scp sd_diffusers.ipynb $USERNAME@euler:/cluster/home/$USERNAME/$DIRECTORY
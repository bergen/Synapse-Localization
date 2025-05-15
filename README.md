## Installation

apt-get install -y pkg-config libcairo2-dev

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -i https://pypi.org/simple -r requirements.txt

pip install flash-attn --no-build-isolation

Much of the code is forked from https://github.com/HazyResearch/hyena-dna

## Weights

Weights for the pretrained model are available [here](https://huggingface.co/lbergen/Synapse-Localization/tree/main).

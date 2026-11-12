# set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# set up hugging face
# for old computer
pip install huggingface_hub==0.36.2

pip install --upgrade huggingface_hub
pip install --upgrade "huggingface_hub[cli]"
hf auth login
# force re-login
hf auth login --force

hf auth whoami

# download a model
hf download google/gemma-3-270m-it
hf download HuggingFaceTB/SmolLM2-135M-Instruct
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct
hf download meta-llama/Llama-3.2-1B-Instruct

# list downloaded models
hf cache ls
hf cache ls --revisions
hf cache ls --sort size:desc

# delete a model
hf cache rm model/google/gemma-3-270m-it --dry-run   # preview first
hf cache rm model/google/gemma-3-270m-it             # actually remove it

# vllm for development
pip install vllm



# set up dependencies
pip install torch
# torch without GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install huggingface_hub==0.36.2
pip install "transformers==4.46.3" "trl==0.12.2" "peft==0.13.2" "accelerate==1.1.1" "datasets==3.1.0" "numpy==1.26.4" "pyarrow==15.0.2" "pandas==2.2.3"



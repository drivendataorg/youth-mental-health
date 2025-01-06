conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
# Note: was previously 
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-cache-dir llama-cpp-python==0.2.90
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install llama-cpp-python --upgrade --force-reinstall
#pip install flash-attn==2.6.1 --no-build-isolation

## Steps of Installing ST-Activity2Vec

 ### Prerequisites
 - Python >= 3.6
 - Numpy
 - PyTorch 1.3.1 + CUDA 9.2
 - TorchVision 0.4.2
 - GCC >= 4.9
 - Anaconda 3(optional)
 - ffmpeg

 ### 1. Create a new conda environment(optional)
 ```
 conda create -y -n st-activity2vec python=3.6
 conda activate st-activity2vec
 conda install pip

 # Set the paths related to cuda version. You may add the following lines to $HOME/.bashrc.
 export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
 export PATH=/usr/local/cuda-9.2/bin:$PATH
 ```

 ### 2. Install the dependencies
 ```
 # Install the related dependencies.
 pip install -r requirements.txt
 
 # Clone this repo.
 git clone https://github.com/DirtyHarryLYL/HAKE-Video
 cd HAKE-Video && git checkout ST-Activity2Vec
  
 # Install pycocotools and pyav.
 pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
 conda install av -c conda-forge
 
 # Install fvcore for pytorch 1.3.1.
 git clone https://github.com/facebookresearch/fvcore.git
 pushd fvcore && git checkout c0ba80ac330ac08dd27ef1ce2e69c455d6f48e56 && popd
 pip install -e fvcore
 
 # Install detectron2 for pytorch 1.3.1.
 git clone https://github.com/facebookresearch/detectron2 detectron2_repo
 pushd detectron2_repo && git reset --hard f4810ac7eb9cd2d1b2c2bfe40f151238523c337c && popd
 pip install -e detectron2_repo
 ```

 ### 3. Setup ST-Activity2Vec
 ```
 python setup.py build develop
 ```

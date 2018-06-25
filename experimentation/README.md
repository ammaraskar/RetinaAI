# Setting up in Miniconda

Doumenting my attempt to use Miniconda to make grabbing the dependencies
easier. https://conda.io/miniconda.html

Download Python 3.* version of Miniconda for Windows.

1. Set up a virtualenv

    * `conda create -n RetinaAI python=3.5.5`

    * `conda activate RetinaAI`

2. Install dependencies

    * `conda install --yes -c conda-forge opencv dlib=19.9 scipy`

    * `pip install Pillow "Click>=6.0" pypiwin32`

    * `pip install --ignore-installed --upgrade tensorflow tflearn`

    * `pip install --no-dependencies face_recognition face_recognition_models`

3. Optionally install GPU accelerated tensorflow
    
    https://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html
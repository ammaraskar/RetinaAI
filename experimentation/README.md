# Setting up in Anaconda

Doumenting my attempt to use Anaconda to make grabbing the dependencies
easier. https://www.anaconda.com/downloads

Download Python 3.* version of Anaconda for Windows.

1. Set up a virtualenv

    * `conda create -n RetinaAI python=3.5.5`

    * `conda activate RetinaAI`

2. Install dependencies

    * `conda install --yes -c conda-forge opencv dlib`

    * `pip install Pillow Click>=6.0`

    * `pip install --no-dependencies face_recognition face_recognition_models`

pip install pypiwin32
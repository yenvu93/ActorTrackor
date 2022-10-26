# ActorTrackor
Evaluation of Actors Tracking Method in Case of Variant Appearance
Introduction

A significant obstacle is unseen films that can not recognize characters in variant appearances. This paper considers using text annotations in form of subtitles and scripts to extract an initial prediction of who appears in the video and when they appear. It is motivated to carry out this challenge by building a deep convolutional neural network to detect pedestrians and automatically detect who is speaking by face detection and lip emotion methodology. Then labeling the correct name of each speaker in the frame using textual semantic cues. Moreover, we also develop a set of results containing actor identification, frame number, and image location which is helpfully extracted the according to actors.


Installation


Ubuntu
The common feature of Linux kernel operating systems is that it requires low hardware configuration, so Ubuntu Desktop is no exception. In this project, Ubuntu 22.04 LTS is used to install the environment. Here is the recommended configuration for the system to run best on the Ubuntu home page:

If researchers install lower versions of Ubuntu, the required configuration may be lower. Moreover, CUDA-enabled GPU hardware requirements to run parallel the CNNs so a GPU that supports the NVIDIA CUDA platform is required.

Anaconda

            conda create --name actortracker python=3.7 -ipython
            
            conda activate actortracker
            
Python
Python 3.6+ or later is required. To avoid conflicts with the NumPy version used in many
dependencies of installing required packages. This project uses Python 3.7 more stability.

CUDA and Pytorch

The followingsteps are guided:

• Checking NVIDIA version by: smi-nvidia. If it is not found, the installation of NVIDIA
on Ubuntu automatically by selecting appreciate NVIDIA version on Additional drivers
in Software and Updates.

• After the above command is performed successfully, the name of recommended CUDA
version is shown. Then going to NVIDIA website to download the appropriate CUDA
version and following steps to install it.

• CUDA is installed successfully under /usr/local/cuda. Then Continuing to install the
PyTorch version by searching on the official Pytorch website. This project use NVIDIA-
driver-470 and CUDA 11.4 and PyTorch 1.10.1.
E.g.1 If have CUDA 10.1 installed under /usr/local/cuda and would like to install PyTorch
1.5, user need to install the prebuilt PyTorch with CUDA 10.1.

            conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

MMCV

Before installing mmcv-full, make sure that PyTorch has been successfully installed following
the official guide. It provides pre-built mmcv packages (recommended) with different PyTorch
and CUDA versions to simplify the building for Linux and Windows systems. The rule for installing the latest mmcv-full is as follows:
For example, to install the latest mmcv-full with CUDA 11.1 and PyTorch 1.9.0, use the fol-lowing command:

            pip3 install mmcv-full==1.1.0 https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

There is an error the developer can face ”the currently installed version of g++ (11.2.0) is greater
than the maximum required version by CUDA 11.4 (10.0.0)”. It is due to the incompatibility
between the GCC/G++ version of the Ubuntu environment with CUDA versions. To fix this
issue, the following stack-overflow website guides:

Check the maximum supported GCC version for your CUDA version: Set an env var for that
GCC version. For example, for CUDA:

            MAX GCC VERSION=8

Make sure that version installed:

            sudo apt install gcc-$MAX GCC VERSION g++-$MAX GCC VERSION

Add symbol link within CUDA folders:

            sudo ln -s /usr/bin/gcc-$MAX GCC VERSION /usr/local/cuda/bin/gcc

            sudo ln -s /usr/bin/g++-$MAX GCC VERSION /usr/local/cuda/bin/g++

MMDetection

MMDetection toolbase is installed the following steps:

• If develop and run mmdet directly, it needs to be clone the code in GitHub:

git clone https://github.com/open-mmlab/mmdetection.git

            pip install -v -e .

• If use mmdet as a dependency or third-party package, install it with pip:

            pip install mmdet

Sentence Transformers

To install Sentence-Transformers, it has to install the dependencies Pytorch and Transformers
first. Pytorch is installed successfully in the above step, install transformers with the command:
pip install transformers
After installing Pytorch and transformers, runt the following command to install Sentence-
Transformers:

            pip install sentence-transformers

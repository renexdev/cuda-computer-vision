## GPU Acceleration of Edge-Based Motion Detection and Machine Learning-Aided Facial Recognition with NVIDIA CUDA


### Background
This is our final course project for the Fall 2015 section of Rice University's [ELEC 301--Signals and Systems](http://dsp.rice.edu/courses/elec301), instructed by [Richard Baraniuk](http://web.ece.rice.edu/richb/).

Our team:
* Emilio Del Vecchio, Electrical and Computer Engineering, Rice University '18
* Kevin Lin, Electrical and Computer Engineering, Rice University '18
* Senthil Natarajan, Electrical and Computer Engineering, Rice University '17

We would also like to thank our mentor CJ Barberan, Rice University ECE PhD student, for his extensive help in our project.


### Abstract
GPUs provide a powerful platform for parallel computations on large data inputs, namely images. In this paper, we explore a GPU-based implementation of a simplified adaptation of existing edge detection algorithms fast enough to operate on frames of a continuous video stream in real-time. We also demonstrate a practical application of edge detection--an edge-based method for motion detection estimation. Additionally, we explore the GPU-CPU speedup of existing OpenCV GPU computation libraries, namely, for facial recognition algorithms. Finally, we demonstrate speedups as high as 10x we achieve with GPU parallelism, as compared to a reference serial CPU-based implementation.

Our full paper is available [here](https://github.com/LINKIWI/cuda-computer-vision/raw/master/documents/paper/paper.pdf), and our poster is available [here](https://github.com/LINKIWI/cuda-computer-vision/raw/master/documents/poster/poster.pdf).


### Prerequisites
1. The NVIDIA CUDA SDK
> Download the latest version of the CUDA development SDK from the [NVIDIA developer website](https://developer.nvidia.com/cuda-downloads) and install it with the [instructions relevant to your platform](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux).

2. OpenCV C and Python libraries
> Obtain the latest build of OpenCV and compile and install it, after meeting all necessary prerequisites.
> ```bash
> sudo apt-get install build-essential
> sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
> sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
> git clone https://github.com/Itseez/opencv.git
> cd opencv
> mkdir release && cd release
> cmake .. && make && sudo make install
> ```

3. Our CUDA source makes use of helper functions referenced in header files in the default CUDA samples directory. In order for our code to compile, simply copy these files to CUDA's default `include` directory.
> ```
> sudo cp /usr/local/cuda/samples/common/inc/helper_* /usr/local/cuda/targets/FOLDER/include/
> ```
> where `FOLDER` is the name of the only directory in `targets/`. It might be, for example, `x86_64`.

4. Python 2.6+ is required for running our Python implementation of facial recognition.
> ```bash
> sudo apt-get install python
> ```


### Running our Code
#### Benchmark (CUDA)
To run our benchmark, simply compile and run the executable. It will run each of our algorithms on the images in the `images/` directory. Note that the benchmark may exit with an error while running if the GPU does not have enough memory to complete the benchmark. (The entire benchmark runs successfully on a GeForce GTX 750 with 2 GB of memory).
```bash
cd cuda/benchmark
make
./benchmark
```

#### Real-time Motion Detection (CUDA)
To run our real-time motion detector, simply compile and run the executable. The program will display 4 windows: the raw input image, the detected edges, the difference map, and the motion area estimation. Note that the program assumes you have a webcam available on your machine.
```bash
cd cuda/main
make
./main
```

#### Facial Recognition, Daft Punk Mask (Python)
This is as simple as running the python script. The program assumes a webcam is available on the machine.
```bash
cd opencv-object-detect
python facedetect.py
```
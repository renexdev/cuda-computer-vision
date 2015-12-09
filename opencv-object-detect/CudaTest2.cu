#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <iostream>
#include <sys/time.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	Ptr<cuda::CascadeClassifier> debugCascade = cuda::CascadeClassifier::create("/home/ubuntu/Downloads/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml");

    if (debugCascade.empty()){
        cerr << "Could not load model!" << endl;
        return -1;
    }

    Mat original = imread("/home/ubuntu/project/opencv-object-detect/senthil1000.jpg");
    Mat gray = imread("/home/ubuntu/project/opencv-object-detect/senthil1000.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat work_img = original.clone();

    // Preprocessing
    cvtColor(original, work_img, COLOR_RGB2GRAY);
    equalizeHist(work_img, work_img);

    bool findLargestObject = false;
    bool filterRects = true;

    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);	
    
    // Detection
    cuda::GpuMat inputGPU(work_img);
    cuda::GpuMat faces;
    debugCascade->setFindLargestObject(findLargestObject);
    debugCascade->setScaleFactor(1.2);
    debugCascade->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
    debugCascade->detectMultiScale(inputGPU, faces);

    gettimeofday(&tend, NULL);
    double runtime = (double) (tend.tv_usec - tstart.tv_usec) / 100000 + (double) (tend.tv_sec - tstart.tv_sec);
    printf("GPU Runtime: %f seconds\n", runtime);
    printf("end %f \n", (double)(tend.tv_usec));
    printf("start %f \n", (double)(tstart.tv_usec));

    vector<Rect> objects;
    debugCascade->convert(faces, objects);

    for(int i=0;i<(int)objects.size();++i)
    {
         rectangle(original, objects[i], Scalar(0, 255, 0), 3);
    }  

    imshow("detections", original);
    moveWindow("detections", 100, 100);
    waitKey(0);
}

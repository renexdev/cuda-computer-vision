#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
//    Ptr<cuda::CascadeClassifier> debugCascade = cuda::CascadeClassifier::create("/home/ubuntu/project/opencv-object-detect/haarcascade_frontalface_default.xml");
	Ptr<cuda::CascadeClassifier> debugCascade = cuda::CascadeClassifier::create("/home/ubuntu/Downloads/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml");

    if (debugCascade.empty()){
        cerr << "Could not load model!" << endl;
        return -1;
    }

    Mat original = imread("/home/ubuntu/project/opencv-object-detect/birthday.jpg");
    Mat gray = imread("/home/ubuntu/project/opencv-object-detect/birthday.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat work_img = original.clone();

    // Preprocessing
    cvtColor(original, work_img, COLOR_RGB2GRAY);
    equalizeHist(work_img, work_img);

    // Detection
    cuda::GpuMat inputGPU(work_img);
    cuda::GpuMat faces;
    debugCascade->detectMultiScale(inputGPU, faces);
    vector<Rect> objects;
    debugCascade->convert(faces, objects);

    for(int i=0;i<(int)objects.size();++i)
    {
         rectangle(original, objects[i], Scalar(255));
    }  

    imshow("detections", original);
    moveWindow("detections", 100, 100);
    waitKey(0);
}

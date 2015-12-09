#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <iostream>
#include <sys/time.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <iomanip>

using namespace std;
using namespace cv;
//using namespace cv::cuda;

int main(int argc, char* argv[])
{
	string cascadeName = "/home/ubuntu/Downloads/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
	Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create(cascadeName);
	cv::CascadeClassifier cascadeCPU;
	cascadeCPU.load(cascadeName);

    if (cascade.empty() || cascadeCPU.empty()){
        cerr << "Could not load model!" << endl;
        return -1;
    }

    string imageName = "/home/ubuntu/project/opencv-object-detect/senthil1000.jpg";
    Mat original = imread(imageName);
    Mat gray = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
    Mat work_img = original.clone();

    // Preprocessing
    cvtColor(original, work_img, COLOR_RGB2GRAY);
    equalizeHist(work_img, work_img);

    bool findLargestObject = false;
    bool filterRects = true;

    struct timeval tstartC, tendC, tstartG, tendG;	
    
    // GPU Detection
    gettimeofday(&tstartG, NULL);
    cuda::GpuMat inputGPU(work_img);
    cuda::GpuMat faces;
    cascade->setFindLargestObject(findLargestObject);
    cascade->setScaleFactor(1.2);
    cascade->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
    cascade->detectMultiScale(inputGPU, faces);

//    gettimeofday(&tendG, NULL);
//    double runtimeG = (double) (tendG.tv_usec - tstartG.tv_usec) / 100000 + (double) (tendG.tv_sec - tstartG.tv_sec);
//    printf("GPU Runtime: %f seconds\n", runtimeG);
    //printf("end %f \n", (double)(tend.tv_usec));
    //printf("start %f \n", (double)(tstart.tv_usec));

    vector<Rect> objects;
    cascade->convert(faces, objects);
   
    for(int i=0;i<(int)objects.size();++i)
    {
         rectangle(original, objects[i], Scalar(0, 255, 0), 3);
    }  
  
    imshow("detections", original);

    gettimeofday(&tendG, NULL);
    double runtimeG = (double) (tendG.tv_usec - tstartG.tv_usec) / 100000 + (double) (tendG.tv_sec - tstartG.tv_sec);
    printf("GPU Runtime: %f seconds\n", runtimeG);

    //CPU Detection
    gettimeofday(&tstartC, NULL);
    cv::Mat inputCPU(work_img);
    //cv::Mat facesCPU;
    vector<Rect> facesCPU;
    Size minsize = cascade->getClassifierSize();
    cascadeCPU.detectMultiScale(inputCPU, facesCPU, 1.1,
                                         (filterRects || findLargestObject) ? 4 : 0,
                                         (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                                            | CASCADE_SCALE_IMAGE,
                                         minsize);

    ///    
    for(int i=0;i<(int)facesCPU.size();++i)
    {
         rectangle(work_img, facesCPU[i], Scalar(0, 255, 0), 3);
    }
    
    imshow("detections2", work_img);

    gettimeofday(&tendC, NULL);
    double runtimeC = (double) (tendC.tv_usec - tstartC.tv_usec) / 100000 + (double) (tendC.tv_sec - tstartC.tv_sec);
    printf("CPU Runtime: %f seconds\n", runtimeC);
    
    double speedup = runtimeC / runtimeG;
    printf("speedup: %f \n", speedup);
 
    //display
//    for(int i=0;i<(int)objects.size();++i)
//    {
//         rectangle(original, objects[i], Scalar(0, 255, 0), 3);
//    }  

//    imshow("detections", original);
    moveWindow("detections", 100, 100);
    waitKey(0);
    moveWindow("detections2", 100, 100);
    waitKey(0);

}

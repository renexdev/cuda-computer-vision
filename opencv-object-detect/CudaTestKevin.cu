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
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

//global call of cascade

void recog(string imageName, string mode)
{
	//string cascadeName = "/home/ubuntu/Downloads/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
        string cascadeName = "haarcascade_frontalface_alt2.xml";
        Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create(cascadeName);
        cv::CascadeClassifier cascadeCPU;
        cascadeCPU.load(cascadeName);

        if (cascade.empty() || cascadeCPU.empty()){
                cerr << "Could not load model!" << endl;
               // return -1;
        }

        Mat original = imread(imageName);
        Mat gray = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        Mat work_img = original.clone();
        Mat copy = original.clone();

        //pre processing
        cvtColor(original, work_img, COLOR_RGB2GRAY);
        equalizeHist(work_img, work_img);

        bool findLargestObject = false;
        bool filterRects = true;

	if (mode == "--GPU")
        {

                cuda::GpuMat inputGPU(work_img);
                cuda::GpuMat faces;
                faces.create(1, 10000, cv::DataType<cv::Rect>::type); ///
                cascade->setFindLargestObject(findLargestObject);
                cascade->setScaleFactor(1.2);
                cascade->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
                cascade->detectMultiScale(inputGPU, faces);

                vector<Rect> objects;
                cascade->convert(faces, objects);

                for(int i=0;i<(int)objects.size();++i)
                {
                     rectangle(original, objects[i], Scalar(0, 255, 0), 3);
                }

                imshow("detections", original);
                //moveWindow("detections", 100, 100);
                //waitKey(0);
        }

        else if (mode == "--CPU")
        {

                cv::Mat inputCPU(work_img);
                vector<Rect> facesCPU;
                Size minsize = cascade->getClassifierSize();
                cascadeCPU.detectMultiScale(inputCPU, facesCPU, 1.1,
                                                     (filterRects || findLargestObject) ? 4 : 0,
                                                     (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                                                        | CASCADE_SCALE_IMAGE,
                                                      minsize);


                for(int i=0;i<(int)facesCPU.size();++i)
                {
                         rectangle(work_img, facesCPU[i], Scalar(0, 255, 0), 3);
                }

                imshow("detections2", copy);
                moveWindow("detections2", 100, 100);
                //waitKey(0);
        }
}

int main(int argc, char* argv[])
{
        const char *images[9];

        images[0] = "senthil100.jpg";
        images[1] = "senthil500.jpg";
        images[2] = "senthil1000.jpg";
        images[3] = "Senthil2000.jpg";
        images[4] = "Senthil3000.jpg";
        images[5] = "Senthil4000.jpg";
        images[6] = "Senthil5000.jpg";
        images[7] = "Senthil6000.jpg";
        images[8] = "Senthil7500.jpg";

        for (int i = 0; i < 9; i++)
        {
                cudaError_t error;

                cudaEvent_t start1;
                error = cudaEventCreate(&start1);

                if (error != cudaSuccess)
                {
                         fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
                         exit(EXIT_FAILURE);
                }

                cudaEvent_t stop1;
                error = cudaEventCreate(&stop1);

                if (error != cudaSuccess)
                {
                         fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
                         exit(EXIT_FAILURE);
                }

                error = cudaEventRecord(start1, NULL);

                if (error != cudaSuccess)
                 {
                        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                recog(string(images[i]), string(argv[1]));

		error = cudaEventRecord(stop1, NULL);
                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                error = cudaEventSynchronize(stop1);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                float msecTotal1 = 0.0f;
                error = cudaEventElapsedTime(&msecTotal1, start1, stop1);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                printf("run1: %.3f msec \n", msecTotal1);


                cudaEvent_t start;
                error = cudaEventCreate(&start);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                cudaEvent_t stop;
                error = cudaEventCreate(&stop);

                if (error != cudaSuccess)
                {
                         fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
                         exit(EXIT_FAILURE);
                }

		error = cudaEventRecord(start, NULL);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                recog(string(images[i]), string(argv[1]));

                error = cudaEventRecord(stop, NULL);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                error = cudaEventSynchronize(stop);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                float msecTotal = 0.0f;
                error = cudaEventElapsedTime(&msecTotal, start, stop);

                if (error != cudaSuccess)
                {
                        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
                        exit(EXIT_FAILURE);
                }

                printf("run2: %.3f msec \n", msecTotal);

                //double speedup = runtimeC / runtimeG;
                //printf("========speedup========= \n");
                //printf("speedup: %f \n", speedup);
        }

        return 0;


}

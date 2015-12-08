#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;

#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>
#include "fstream"
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/cuda.hpp>

//cv::cuda::CascadeClassifier_CUDA cascade_gpu;

Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create("haarcascade_frontalface_default.xml");

//vector<Rect> detect_faces(Mat& image)
void detect_faces(Mat& image)
{
	vector<Rect> res;
	bool findLargestObject = true;
	bool filterRects = true;
	int detections_num;
	Mat faces_downloaded;
	Mat im(image.size(), CV_8UC1);
	GpuMat facesBuf_gpu;
	if(image.channels()==3)
	{
		cvtColor(image, im, CV_BGR2GRAY);
	}
	else
	{
		image.copyTo(im);
	}
	GpuMat gray_gpu(im);
	
//	cascade_gpu.visualizeInPlace = false;
//	findLargestObject=cascade_gpu->FindLargestObject;
//	detections_num = cascade_gpu->detectMultiScale(gray_gpu, facesBuf_gpu, 1.2, (filterRects || findLargestObject) ? 4 : 0,Size(image.cols/4,image.rows/4));
//	detections_num = cascade_gpu->detectMultiScale(gray_gpu, facesBuf_gpu);
//	if(detections_num==0){return res;}
	
//	facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
//	Rect *faceRects = faces_downloaded.ptr<Rect>();

//	for(int i = 0; i<detections_num; i++)
//	{
//		res.push_back(faceRects[i]);
//	}
	
	cascade_gpu->detectMultiScale(gray_gpu, facesBuf_gpu);
	std::vector<Rect> faces;
	cascade_gpu->convert(facesBuf_gpu, faces);

	for (int i = 0; i < faces.size(); ++i)
		cv::rectangle(image, faces[i], Scalar(255));

	imshow("Faces", image);

//	gray_gpu.release();
//	facesBuf_gpu.release();
//	return res;
		
}


int main()
{
  //  printf("\n Hello World\n");
  //  return EXIT_SUCCESS;
	
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
//	cascade_gpu.load("haarcascade_frontalface_default.xml");
	Mat frame, img;
	namedWindow("frame");
	VideoCapture capture(0);
	capture >> frame;
	vector<Rect> rects;
	if (capture.isOpened())
	{
		while(waitKey(20)!=27)
		{
			capture >> frame;
			cvtColor(frame, img, CV_BGR2GRAY);
			detect_faces(img);
		//	rects=detect_faces(img);
		//	if(rects.size()>0)
		//	{
		//		cv::rectangle(frame, rects[0], CV_RGB(255,0,0));
		//	}
		//	imshow("frame", frame);
		}
	}

	return 0;

}



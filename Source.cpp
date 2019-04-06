#include "windows.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<sstream>
#include<fstream>

using namespace std;
using namespace cv;
using namespace dnn;
float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;
vector<string> classes2;

vector<string> classes;
string address = "C:/Users/mahyar/Desktop/opencv projects";



vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers();
		vector<String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	string label = format("%.2f", conf);
	if (!classes2.empty())
	{
		CV_Assert(classId < (int)classes2.size());
		label = classes2[classId] + ":" + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}


void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
	drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}





int main() {

	/*string file = "C:/Users/mahyar/Desktop/opencv projects/classification_classes_ILSVRC2012.txt";
	ifstream ifs(file.c_str());
	if (!ifs.is_open()) {
	
		cout << "couldn't open classes files";
	
	}
	else {
	
		string line;
		while (getline(ifs,line)) {
		
			classes.push_back(line);
			

			cout << "pushed back: " <<endl;
		
		}
	
	}*/
	/*Net net = readNet("C:/Users/mahyar/Desktop/opencv projects/bvlc_googlenet.caffemodel", "C:/Users/mahyar/Desktop/opencv projects/bvlc_googlenet.prototxt");

	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
*/

	//Net net = readNet("G:/Science/My papers/AI/Object detction/yolo-object-detection/yolo-coco/yolov3.weights", "G:/Science/My papers/AI/Object detction/yolo-object-detection/yolo-coco/yolov3.cfg");
	//
	//Mat img;
	//img = imread("C:/Users/mahyar/Desktop/opencv projects/space_shuttle.jpg");
	//



	VideoCapture cap;
	cap.open(address + "/Adriana.MOV");
	//cap.open(0);





	Mat frame,blob;
	int inpWidth = 224;
	int inpHeight = 224;
	Scalar mean = (104,117,123);
	int frame_counter = 0;
	int scale = 1;
	//test the shuttle
	/*blobFromImage(img, blob, scale, Size(inpWidth, inpHeight), mean, true, false);
	net.setInput(blob);
	Mat prob = net.forward();
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time: %.2f ms", t);
	putText(img, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
		classes[classId].c_str()),
		confidence);
	putText(img, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	imshow("shuttle", img);

*/


	//while (waitKey(1)<0) {
	//	cout << frame_counter<<endl;
	//	cap >> frame;
	//	
	//	if (frame.empty())
	//	{
	//		waitKey();
	//		break;

	//	}
	////	imshow("original", frame);
	//	blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, true, false);
	//	net.setInput(blob);
	//	Mat prob = net.forward();

	//	Point classIdPoint;
	//	double confidence;
	//	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	//	int classId = classIdPoint.x;

	//	vector<double> layersTimes;
	//	double freq = getTickFrequency() / 1000;
	//	double t = net.getPerfProfile(layersTimes) / freq;
	//	string label = format("Inference time: %.2f ms", t);
	//	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

	//	label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
	//		classes[classId].c_str()),
	//		confidence);
	//	putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));


	//	imshow("result", frame);



	//	frame_counter++;
	//}















	//yolo v3

	
	string classesFile = "G:/Science/My papers/AI/Object detction/yolo-object-detection/yolo-coco/coco.names";
	ifstream ifs2(classesFile.c_str());
	string line;
	while (getline(ifs2, line)) classes2.push_back(line);
	String modelConfiguration = "G:/Science/My papers/AI/Object detction/yolo-object-detection/yolo-coco/yolov3.cfg";
	String modelWeights = "G:/Science/My papers/AI/Object detction/yolo-object-detection/yolo-coco/yolov3.weights";
	Net net2 = readNetFromDarknet(modelConfiguration, modelWeights);
	net2.setPreferableBackend(DNN_BACKEND_OPENCV);
	net2.setPreferableTarget(DNN_TARGET_CPU);
	frame_counter = 0;
	while (waitKey(1) < 0)
	{
		cout << frame_counter<<endl;
		cap >> frame;
		if (frame.empty()) {
			cout << "Done processing" << endl;
			waitKey(3000);
			break;
		}

		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		net2.setInput(blob);

		vector<Mat> outs;
		net2.forward(outs, getOutputsNames(net2));

		postprocess(frame, outs);

		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net2.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		frame_counter++;
		imshow("frame", frame);

	}



















	//string address = "C:/Users/mahyar/Desktop/opencv projects";

	////VideoCapture video(address+"/Adriana.MOV");
	//VideoCapture video(0);
	//Ptr<BackgroundSubtractor> bs = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();


	//if (!video.isOpened()) {
	//
	//
	//	cout << "Error reading video";
	//	return 0;
	//}


	//int frame_counter = 0;
	////frame_counter is for counting the current frame number
	//Mat frame, fgmask, background,foreground_img;

	//while (1) {
	//	cout << frame_counter<<endl;
	//	frame_counter++;
	//	video >> frame;

	//	



	//	if (frame.empty()) {
	//	
	//		cout << "There is no frame";
	//		break;
	//	}
	//	//imshow("frame", frame);
	//	//cvtColor(frame, frame, COLOR_BGR2GRAY);
	//	resize(frame, frame, Size(1000, 500));
	//		if (fgmask.empty()) {
	//			fgmask.create(frame.size(), frame.type());
	//		}
	//		//imshow("foreground mask", fgmask);
	//		bs->apply(frame, fgmask, true ? -1 : 0);
	//	//	GaussianBlur(fgmask, fgmask, Size(3, 3), 3.5, 3.5);
	//	//	imshow("foreground mask", fgmask);	
	//		threshold(fgmask, fgmask, 100, 255.0, THRESH_BINARY);
	//		//imshow("foreground after threshold", fgmask);
	//		foreground_img = Scalar::all(0);
	//		frame.copyTo(foreground_img, fgmask);
	//		bs->getBackgroundImage(background);
	//		/*if (!background.empty()) {

	//			imshow("mean background image", background);
	//			int key5 = waitKey(40);

	//		}*/
	//		namedWindow("foreground_img", WINDOW_FULLSCREEN);
	//		//imshow("frame", frame);
	//		//imshow("backgound", background);
	//		imshow("foreground_img", foreground_img);
	//		/*Mat  diff;
	//		absdiff(background, foreground_img, diff);
	//		imshow("diffrence", diff);*/
	//		stringstream s;
	//		s << fgmask.size() << frame.size();
	//		/*string ss = s.str();
	//		cout << ss << endl;*/
	//	char c = (char)waitKey(25);
	//}
	//cap.release();
	//destroyAllWindows();
	//cout <<endl<<"this is my address"<< address+"/Adriana.MOV";



	
//	cout << "sdlcnhsdhvbhjsdjkvdskvbdhvhsdjbvjdsbvjhdbhjvbdv";
//	try {
//		Mat img = imread("jacket.jpg");
		
//		namedWindow("image", WINDOW_NORMAL);
	//	imshow("image", img);
		//waitKey(0);
	
	
	//}
	//catch (Exception & e) {
		
	
		//cerr << "we have an error here *********************************************" << e.msg<<"\n"<<"we have an error here *********************************************";
	//}

	return 0;
}













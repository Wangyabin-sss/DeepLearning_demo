#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


string modelsFile = "./lenet.onnx";
string imageFile = "../images/2.jpg";


int main(int argc, char **argv)
{

    Scalar mean(0.1307);
	Scalar std(0.3081);

    Mat frame = imread(imageFile);
	cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
    resize(frame,frame,Size(28,28));
	frame.convertTo(frame, CV_32FC1, 1.0f/255.0f);
    cout<<frame.size<<endl;

    imshow("mat0",frame);

	frame = (frame - mean)/std;
    Net net = readNetFromONNX(modelsFile);
    Mat inpBlob = blobFromImage(frame, 1.0, Size(28, 28), Scalar(0), true, true);

    net.setInput(inpBlob);
    Mat out = net.forward();
    cout<<out.size<<" depth= "<<out.depth()<<endl;

    float *data = (float*)out.data;

	for(size_t i=0;i<10;i++)
    {
        std::cout<<data[i]<<"  \t"<<i<<"  \t"<<exp(data[i])<<std::endl;
    }

    waitKey(0);

    return 0;
}







#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Read the image file
    Mat image = imread("media/DraughtsGame1.JPG");
    // Check for failure
    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    // Print image values to console.
    int total = image.total();
    printf("Number of pixels = %d\n", total);


    // Take samples for all four possible pieces
    // Backproject all of them

    // Show our image inside a window.
    //imshow("Image Window Name here", image);

    // Wait for any keystroke in the window
    waitKey(0);
    return 0;
}





/*
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat hue;
int bins = 25;
void Hist_and_Backproj(int, void*);
int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, "{@input |  | input image}");
    //Mat src = imread(parser.get<String>("@input"));
    Mat src = imread(parser.get<String>("media/DraughtsGame1.JPG"));
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    hue.create(hsv.size(), hsv.depth());
    int ch[] = { 0, 0 };
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
    const char* window_image = "Source image";
    namedWindow(window_image);
    createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj);
    Hist_and_Backproj(0, 0);
    imshow(window_image, src);
    // Wait until user exits the program
    waitKey();
    return 0;
}
void Hist_and_Backproj(int, void*)
{
    int histSize = MAX(bins, 2);
    float hue_range[] = { 0, 180 };
    const float* ranges[] = { hue_range };
    Mat hist;
    calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
    normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
    Mat backproj;
    calcBackProject(&hue, 1, 0, hist, backproj, ranges, 1, true);
    imshow("BackProj", backproj);
    int w = 400, h = 400;
    int bin_w = cvRound((double)w / histSize);
    Mat histImg = Mat::zeros(h, w, CV_8UC3);
    for (int i = 0; i < bins; i++)
    {
        rectangle(histImg, Point(i * bin_w, h), Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)),
            Scalar(0, 0, 255), FILLED);
    }
    imshow("Histogram", histImg);
}*/



/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
std::string window_name = "Threshold Demo";

std::string trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
std::string trackbar_value = "Value";

// Function headers
void Threshold_Demo(int, void*);

int main(int argc, char** argv)
{
    /// Load an image
    src = imread("media/DraughtsGame1.JPG");

    /// Convert the image to Gray
    cvtColor(src, src_gray, cv::COLOR_BGRA2GRAY);

    /// Create a window to display results
    namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    /// Create Trackbar to choose type of Threshold
    createTrackbar(trackbar_type,
        window_name, &threshold_type,
        max_type, Threshold_Demo);

    createTrackbar(trackbar_value,
        window_name, &threshold_value,
        max_value, Threshold_Demo);

    /// Call the function to initialize
    Threshold_Demo(0, 0);

    /// Wait until user finishes program
    while (true)
    {
        int c;
        c = waitKey(20);
        if ((char)c == 27)
        {
            break;
        }
    }

}


void Threshold_Demo(int, void*)
{
    threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

    imshow(window_name, dst);
}*/
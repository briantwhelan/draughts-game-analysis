#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Namespaces.
using namespace cv;
using namespace std;

// Function definitions.
void PrintMatrix(string name, Mat matrix, int limit);
Mat ExtractHue(Mat rgb_image);
Mat HueHistogram(Mat image, int bins);
void DisplayHistogram(string name, Mat hist, int bins);
Mat Backproject(int, void*, Mat image, Mat* hue);
void HistogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins);
void DisplayImage(string name, Mat image);

void Test();

int main(int argc, char** argv)
{
    //MyApplication();
    Test();

    // Wait for any keystroke in the window
    waitKey(0);
    return 0;
}

void Test()
{
   // Take samples for all four possible pieces
   // Backproject all of them

   // Read the image files.
   // Mat rgb_image = imread("Media/DraughtsGame1EmptyBoard.JPG");
   Mat rgb_image = imread("Media/DraughtsGame1Move0.JPG");
   Mat white_squares_image = imread("Media/DraughtsGame1WhiteSquares.jpg");
   Mat black_squares_image = imread("Media/DraughtsGame1BlackSquares.jpg");
   Mat white_pieces_image = imread("Media/DraughtsGame1WhitePieces.jpg");
   Mat black_pieces_image = imread("Media/DraughtsGame1BlackPieces.jpg");

   // Check for failures.
   if (rgb_image.empty() || white_squares_image.empty() || black_squares_image.empty() || white_pieces_image.empty() || black_pieces_image.empty())
   {
       cout << "Image Not Found!!!" << endl;
       cin.get(); //wait for any key press
       return;
   }

   // Print RGB image.
   PrintMatrix("RGB", rgb_image, 10);

   // Show source image.
   DisplayImage("source image", rgb_image);

   // Backproject.
   int bins = 25;
   HistogramAndBackproject("white squares", white_squares_image, rgb_image, bins);
   HistogramAndBackproject("black squares", black_squares_image, rgb_image, bins);
   HistogramAndBackproject("white pieces", white_pieces_image, rgb_image, bins);
   HistogramAndBackproject("black pieces", black_pieces_image, rgb_image, bins);
}

// Print the given matrix (ith an upper limit of elements to print).
void PrintMatrix(string name, Mat matrix, int limit) {
    cout << name << ":" << endl;
    cout << "Rows = " << matrix.rows << "\tCols = " << matrix.cols << endl;
    for (int i = 0; limit > 0 && i < matrix.rows; i++)
    {
        for (int j = 0; limit > 0 && j < matrix.cols; j++)
        {
            cout << matrix.at<Vec3b>(i, j);
            limit--;
        }
        cout << endl;
    }
}

// Extract Hue channel from RGB image converted to HSV image.
Mat ExtractHue(Mat rgb_image)
{
    // Convert RGB to HSV.
    Mat hue;
    Mat hsv_image;
    cvtColor(rgb_image, hsv_image, COLOR_BGR2HSV);
    //DisplayImage("HSV", hsv_image);

    // Extract hue channel.
    hue.create(hsv_image.size(), hsv_image.depth());
    int ch[] = { 0, 0 };
    mixChannels(&hsv_image, 1, &hue, 1, ch, 1);
    //DisplayImage("Hue", hue);

    return hue;
}

// Create a hue histogram from the provided RGB image.
Mat HueHistogram(Mat rgb_image, int bins)
{
    // Extract hue channel.
    Mat hue = ExtractHue(rgb_image);
    
    // Create histogram.
    int histSize = MAX(bins, 2);
    float hue_range[] = { 0, 180 };
    const float* ranges[] = { hue_range };
    Mat hist;
    calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

    // Normalise histogram.
    normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

    return hist;
}

void DisplayHistogram(string name, Mat hist, int bins)
{
    // Display histogram.
    int w = 400, h = 400;
    int histSize = MAX(bins, 2);
    int bin_w = cvRound((double)w / histSize);
    Mat histImg = Mat::zeros(h, w, CV_8UC3);
    for (int i = 0; i < bins; i++)
    {
        rectangle(histImg, Point(i * bin_w, h), Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)),
            Scalar(0, 0, 255), FILLED);
    }
    imshow(name, histImg);
}

// Backproject the given histogram onto the image.
Mat Backproject(int, void*, Mat hist, Mat* image)
{
    // Backproject the histogram onto image.
    Mat backproj;
    float hue_range[] = { 0, 180 };
    const float* ranges[] = { hue_range };
    calcBackProject(image, 1, 0, hist, backproj, ranges, 1, true);
    return backproj;
}

// Perform backprojection.
void HistogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins) {
    // Histogram hue in sample image.
    Mat histogram = HueHistogram(sample_image, bins);

    // Print histogram.
    cout << name << " histogram:\n" << histogram << endl;

    // Show histogram.
    // DisplayHistogram(name, histogram, bins);

    // Extract Hue channel from RGB image.
    Mat image = ExtractHue(rgb_image);

    // Backproject
    Mat backprojection = Backproject(0, 0, histogram, &image);

    // Print backprojection.
    PrintMatrix(name + " backprojection", backprojection, 100);

    // Show backprojection.
    DisplayImage(name + " backprojection", backprojection);
}

// Display the given image with the provided name.
void DisplayImage(string name, Mat image)
{
    imshow(name, image);
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
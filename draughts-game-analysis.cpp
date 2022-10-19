#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image);
Mat BackProject(Mat image);
Mat ColourHistogram(Mat image);
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

   // Read the image file
   Mat image = imread("Media/DraughtsGame1EmptyBoard.JPG");

   // Check for failure
   if (image.empty())
   {
       cout << "Image Not Found!!!" << endl;
       cin.get(); //wait for any key press
       return;
   }

   // Histogram colours in image.
   Mat colour_histogram = ColourHistogram(image);

   // Display colour histogram.
   imshow("Colour Histogram", colour_histogram);

   //// Backproject
   //Mat backproject = BackProject(image);

   //// Display backprojection.
   //imshow("Backprojection", backproject);

   //// Print image values to console.
   //int total = image.total();
   //printf("Number of pixels = %d\n", total);
}

void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
{
    int number_of_bins = histograms[0].size[0];
    double max_value = 0, min_value = 0;
    double channel_max_value = 0, channel_min_value = 0;
    for (int channel = 0; (channel < number_of_histograms); channel++)
    {
        minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
        max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
        min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
    }
    float scaling_factor = ((float)256.0) / ((float)number_of_bins);

    Mat histogram_image((int)(((float)number_of_bins) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
    display_image = histogram_image;
    line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
    line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
    int highest_point = static_cast<int>(0.9 * ((float)number_of_bins) * scaling_factor);
    for (int channel = 0; (channel < number_of_histograms); channel++)
    {
        int last_height;
        for (int h = 0; h < number_of_bins; h++)
        {
            float value = histograms[channel].at<float>(h);
            int height = static_cast<int>(value * highest_point / max_value);
            int where = (int)(((float)h) * scaling_factor);
            if (h > 0)
                line(histogram_image, Point((int)(((float)(h - 1)) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - last_height),
                    Point((int)(((float)h) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - height),
                    Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
            last_height = height;
        }
    }
}

Mat ColourHistogram(Mat image)
{
    //// Just so that tests can be done using a grayscale image...
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    // Greyscale and Colour (RGB) histograms
    Mat gray_histogram_display_image;
    MatND gray_histogram;
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 64;
    calcHist(&gray_image, 1, channel_numbers, Mat(), gray_histogram, 1, &number_bins, &channel_ranges);
    Draw1DHistogram(&gray_histogram, 1, gray_histogram_display_image);
    return gray_histogram_display_image;

   /* Mat colour_display_image;
    MatND* colour_histogram = new MatND[image.channels()];
    vector<Mat> colour_channels(image.channels());
    split(image, colour_channels);
    for (int chan = 0; chan < image.channels(); chan++)
        calcHist(&(colour_channels[chan]), 1, channel_numbers, Mat(),
            colour_histogram[chan], 1, &number_bins, &channel_ranges);
    OneDHistogram::Draw1DHistogram(colour_histogram, image.channels(), colour_display_image);
    Mat gray_fruit_image_display;
    cvtColor(gray_fruit_image, gray_fruit_image_display, COLOR_GRAY2BGR);
    Mat output1 = JoinImagesHorizontally(gray_fruit_image_display, "Grey scale image", gray_histogram_display_image, "Greyscale histogram", 4);
    Mat output2 = JoinImagesHorizontally(fruit_image, "Colour image", colour_display_image, "RGB Histograms", 4);
    Mat output3 = JoinImagesHorizontally(output1, "", output2, "", 4);
    imshow("Histograms", output3);*/
}

Mat BackProject(Mat image)
{
    Mat backprojection = image.clone();
    float channelRange[2] = { 0.0, 255.0 };
    MatND histogram[3];
    //if (image.channels() == 1)
    //{
       // const float* channel_ranges[] = { channelRange, channelRange, channelRange };
        //for (int channel = 0; (channel < image.channels()); channel++)
        //{
            //calcBackProject(&image, 1, image.channels(), *histogram, backprojection, channel_ranges, 255.0);
        //}
    //}
    //else
    //{
    //}
    return backprojection;
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
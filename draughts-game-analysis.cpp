#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Read the image file
    Mat image = imread("C:/Users/brian/source/repos/opencvtest/B.png");
    // Check for failure
    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press
        return -1;
    }
    // Show our image inside a window.
    imshow("Image Window Name here", image);

    // Wait for any keystroke in the window
    waitKey(0);
    return 0;
}
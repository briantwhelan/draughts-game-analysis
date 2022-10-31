#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
#include <regex>
#include <string>
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

// Function definitions.
void part1(Mat black_pieces_image, Mat white_pieces_image, Mat black_squares_image, Mat white_squares_image);
void part2(Mat empty_board_image, int confusion_matrix[3][3]);
void part3();
void part4();
void part5();

void PrintMatrix(string name, Mat matrix, int limit);
void DisplayImage(string name, Mat image);
int GetObjectPixelsInImage(Mat binary_image);

Mat PerspectiveTransformation(Mat board_image);
Mat GetStructuringElement3x3();
Mat GetStructuringElement5x5();

Mat Erode(Mat binary_image, Mat structuring_element);
Mat Dilate(Mat binary_image, Mat structuring_element);
Mat Opening(Mat binary_image, Mat structuring_element);
Mat Closing(Mat binary_image, Mat structuring_element);

bool isBlackSquare(int top_left_x, int top_left_y);
int GetNumberOfObjectPixelsInSquare(Mat binary_image, int top_left_x, int top_left_y);
bool isPieceInSquare(Mat binary_image, int top_left_x, int top_left_y);
bool isBlackPiece(Mat rgb_image, int top_left_x, int top_left_y);
int CheckBoardGroundTruth(int square_number, string white_pieces, string black_pieces);
void UpdateConfusionMatrix(int confusion_matrix[3][3], int detected_square_contents, int actual_square_contents);


Mat ExtractHue(Mat rgb_image);
Mat HueHistogram(Mat image, int bins);
void DisplayHistogram(string name, Mat hist, int bins);
//Mat Backproject(int, void*, Mat image, Mat* hue);
//void HistogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins);


// Data provided:  Filename, White pieces, Black pieces
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the images.
const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
	{"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
	{"DraughtsGame1Move64.JPG", "4,14", "K2,11,19,20,22"},
	{"DraughtsGame1Move65.JPG", "4,18", "K2,11,19,20,22"},
	{"DraughtsGame1Move66.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move67.JPG", "8", "K2,11,15,19,20"},
	{"DraughtsGame1Move68.JPG", "", "K2,K4,15,19,20"}
};

// Data provided:  Approx. frame number, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const int GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[][3] = {
{ 17, 9, 13 },
{ 37, 24, 20 },
{ 50, 6, 9 },
{ 65, 22, 17 },
{ 85, 13, 22 },
{ 108, 26, 17 },
{ 123, 9, 13 },
{ 161, 30, 26 },
{ 180, 13, 22 },
{ 201, 25, 18 },
{ 226, 12, 16 },
{ 244, 18, 14 },
{ 266, 10, 17 },
{ 285, 21, 14 },
{ 308, 2, 6 },
{ 326, 26, 22 },
{ 343, 6, 9 },
{ 362, 22, 18 },
{ 393, 11, 15 },
{ 433, 18, 2 },
{ 453, 9, 18 },
{ 472, 23, 14 },
{ 506, 3, 7 },
{ 530, 20, 11 },
{ 546, 7, 16 },
{ 582, 2, 7 },
{ 617, 8, 11 },
{ 641, 27, 24 },
{ 673, 1, 6 },
{ 697, 7, 2 },
{ 714, 6, 9 },
{ 728, 14, 10 },
{ 748, 9, 14 },
{ 767, 10, 7 },
{ 781, 14, 17 },
{ 801, 7, 3 },
{ 814, 11, 15 },
{ 859, 24, 20 },
{ 870, 16, 19 },
{ 891, 3, 7 },
{ 923, 15, 18 },
{ 936, 7, 10 },
{ 955, 18, 22 },
{ 995, 10, 14 },
{ 1014, 17, 21 },
{ 1034, 14, 17 },
{ 1058, 21, 25 },
{ 1075, 17, 26 },
{ 1104, 25, 30 },
{ 1129, 31, 27 },
{ 1147, 30, 23 },
{ 1166, 27, 18 },
{ 1182, 19, 23 },
{ 1201, 18, 15 },
{ 1213, 23, 26 },
{ 1243, 15, 11 },
{ 1266, 26, 31 },
{ 1280, 32, 27 },
{ 1298, 31, 24 },
{ 1324, 28, 19 },
{ 1337, 5, 9 },
{ 1358, 29, 25 },
{ 1387, 9, 14 },
{ 1450, 25, 15 },
{ 1465, 4, 8 },
{ 1490, 11, 4 }
};

#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)

#define BOARD_DIMENSIONS_IN_PIXELS 400
#define SQUARE_DIMENSIONS_IN_PIXELS (BOARD_DIMENSIONS_IN_PIXELS/NUMBER_OF_SQUARES_ON_EACH_SIDE)
#define PIXELS_IN_SQUARE (SQUARE_DIMENSIONS_IN_PIXELS*SQUARE_DIMENSIONS_IN_PIXELS)
#define NUMBER_OF_STATIC_IMAGES 69

class DraughtsBoard
{
private:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	Mat mOriginalImage;
	void loadGroundTruth(string pieces, int man_type, int king_type);
public:
	DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth);
};

DraughtsBoard::DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth)
{
	for (int square_count = 1; square_count <= NUMBER_OF_SQUARES; square_count++)
	{
		mBoardGroundTruth[square_count - 1] = EMPTY_SQUARE;
	}
	loadGroundTruth(white_pieces_ground_truth, WHITE_MAN_ON_SQUARE, WHITE_KING_ON_SQUARE);
	loadGroundTruth(black_pieces_ground_truth, BLACK_MAN_ON_SQUARE, BLACK_KING_ON_SQUARE);
	string full_filename = "Media/" + filename;
	mOriginalImage = imread(full_filename, -1);
	if (mOriginalImage.empty())
		cout << "Cannot open image file: " << full_filename << endl;
	else imshow(full_filename, mOriginalImage);
}

void DraughtsBoard::loadGroundTruth(string pieces, int man_type, int king_type)
{
	int index = 0;
	while (index < pieces.length())
	{
		bool is_king = false;
		if (pieces.at(index) == 'K')
		{
			is_king = true;
			index++;
		}
		int location = 0;
		while ((index < pieces.length()) && (pieces.at(index) >= '0') && (pieces.at(index) <= '9'))
		{
			location = location * 10 + (pieces.at(index) - '0');
			index++;
		}
		index++;
		if ((location > 0) && (location <= NUMBER_OF_SQUARES))
			mBoardGroundTruth[location - 1] = (is_king) ? king_type : man_type;
	}
}





void MyApplication()
{
	string video_filename("Media/DraughtsGame1.avi");
	VideoCapture video;
	video.open(video_filename);

	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);
	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, -1);
	if ((!video.isOpened()) || (black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty())  || (static_background_image.empty()))
	{
		// Error attempting to load something.
		if (!video.isOpened())
			cout << "Cannot open video file: " << video_filename << endl;
		if (black_pieces_image.empty())
			cout << "Cannot open image file: " << black_pieces_filename << endl;
		if (white_pieces_image.empty())
			cout << "Cannot open image file: " << white_pieces_filename << endl;
		if (black_squares_image.empty())
			cout << "Cannot open image file: " << black_squares_filename << endl;
		if (white_squares_image.empty())
			cout << "Cannot open image file: " << white_squares_filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		// Classify the pixels.
		//part1(black_pieces_image, white_pieces_image, black_squares_image, white_squares_image);

		// Compute confusion matrix for pieces in squares.
		int confusion_matrix[3][3] = { {0, 0, 0}, 
									   {0, 0, 0}, 
									   {0, 0, 0} };
		part2(static_background_image, confusion_matrix);
		cout << "Confusion Matrix:\n"
			<< "\tGT_NP\tGT_WP\tGT_BP\n"
			<< "D_NP\t" << confusion_matrix[0][0] << "\t" << confusion_matrix[0][1] << "\t" << confusion_matrix[0][2] << "\n" 
			<< "D_WP\t" << confusion_matrix[1][0] << "\t" << confusion_matrix[1][1] << "\t" << confusion_matrix[1][2] << "\n" 
			<< "D_BP\t" << confusion_matrix[2][0] << "\t" << confusion_matrix[2][1] << "\t" << confusion_matrix[2][2] << endl;

		// Record moves in video.
		//part3();

		// Identify four corners of the chessboard.
		//part4();

		// Distinguish between normal pieces and kings.
		//part5();

		//// Sample loading of image and ground truth
		//int image_index = 21;
		//DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);

		//// Process video frame by frame
		//Mat current_frame;
		//video.set(cv::CAP_PROP_POS_FRAMES, 1);
		//video >> current_frame;
		//double last_time = static_cast<double>(getTickCount());
		//double frame_rate = video.get(cv::CAP_PROP_FPS);
		//double time_between_frames = 1000.0 / frame_rate;
		//while (!current_frame.empty())
		//{
		//	double current_time = static_cast<double>(getTickCount());
		//	double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
		//	int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
		//	last_time = current_time;
		//	imshow("Draughts video", current_frame);
		//	video >> current_frame;
		//	char c = cv::waitKey(delay);  // If you replace delay with 1 it will play the video as quickly as possible.
		//}
		//cv::destroyAllWindows();
	}
}

void part1(Mat black_pieces_image, Mat white_pieces_image, Mat black_squares_image, Mat white_squares_image)
{
	// Load classification image.
	string classification_filename = "Media/DraughtsGame1Move0.JPG";
	Mat classification_image = imread(classification_filename, -1);
	if (classification_image.empty())
	{
		cout << "Cannot open image file: " << classification_filename << endl;
	}

	// Load ground truth image.
	string ground_truth_filename = "Media/DraughtsGame1Move0GroundTruth.JPG";
	Mat ground_truth_image = imread(ground_truth_filename, -1);
	if (ground_truth_image.empty())
	{
		cout << "Cannot open image file: " << ground_truth_filename << endl;
	}

	// Print RGB image.
	PrintMatrix("RGB", classification_image, 10);

	// Show source image.
	DisplayImage("Image to classify", classification_image);

	// Backproject.
	int bins = 25;
	//Mat wp = BackProjection(classification_image, white_pieces_image);
	Mat bp = BackProjection(classification_image, black_pieces_image);
	Mat binary_image;
	threshold(bp, binary_image, 127, 255, THRESH_BINARY);
	DisplayImage("binary", binary_image);
	Mat eroded_image = Erode(binary_image, GetStructuringElement3x3());
	DisplayImage("E", eroded_image);
	Mat dilated_image = Dilate(binary_image, GetStructuringElement5x5());
	DisplayImage("D", dilated_image);

	//Mat eroded_image2 = Erode(dilated_image);
	////DisplayImage("DE", eroded_image2);
	//Mat eroded_image3 = Erode(eroded_image2);
	//DisplayImage("DEE", eroded_image3);
	//Mat eroded_image4 = Dilate(eroded_image3);
	//DisplayImage("DEED", eroded_image4);
	//Mat eroded_image5 = Dilate(eroded_image4);
	//DisplayImage("DEEDD", eroded_image5);
	//Mat dilated_image2 = Dilate(eroded_image);
	////DisplayImage("ED", dilated_image2);
	/*Mat o = Opening(binary_image);
	DisplayImage("O", o);
	Mat c = Closing(binary_image);
	DisplayImage("C", c);*/

	/*Mat o2 = Opening(c);
	DisplayImage("O2", o2);
	Mat c2 = Closing(o);
	DisplayImage("C2", c2);

	Mat c3 = Closing(o2);
	DisplayImage("C3", c3);

	Mat c4 = Dilate(c3);
	DisplayImage("C4", c4);*/

	//Mat ws = BackProjection(classification_image, white_squares_image);
	//Mat bs = BackProjection(classification_image, black_squares_image);
	//Mat p = JoinImagesHorizontally(wp, "white pieces", bp, "black pieces", 4, 255);
	//Mat s = JoinImagesHorizontally(ws, "white squares", bs, "black squares", 4, 255);
	//Mat backprojections = JoinImagesVertically(p, "", s, "", 4, 255);
	//DisplayImage("backprojections", backprojections);
	//HistogramAndBackproject("white pieces", white_pieces_image, classification_image, bins);
	//HistogramAndBackproject("black pieces", black_pieces_image, classification_image, bins);
	//HistogramAndBackproject("white squares", white_squares_image, classification_image, bins);
	//HistogramAndBackproject("black squares", black_squares_image, classification_image, bins);
	//HistogramAndBackproject("black pieces2", black_pieces_image2, classification_image, bins);
}

void part2(Mat empty_board_image, int confusion_matrix[3][3])
{
	// Perform perspective transformation on empty board.
	Mat empty_board_pt = PerspectiveTransformation(empty_board_image);
	
	// Otsu Threshold empty board image.
	Mat grey_board_image;
	cvtColor(empty_board_pt, grey_board_image, COLOR_BGR2GRAY);
	Mat thresholded_board_image;
	threshold(grey_board_image, thresholded_board_image, 127, 255, THRESH_BINARY | THRESH_OTSU);
	//DisplayImage("Thresholded Board Image", thresholded_board_image);

	// Tidy with opening to get binary empty board.
	Mat binary_empty_board = Opening(thresholded_board_image, GetStructuringElement5x5());
	//DisplayImage("Binary Empty Board", binary_empty_board);

	// Process all 67 static images.
	for (int image_index = 0; image_index < NUMBER_OF_STATIC_IMAGES; image_index++)
	{
		cout << "Image " << image_index << endl;
		// Load current board image.
		string filename = "Media/" + GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0];
		Mat current_board_image = imread(filename, -1);
		if (current_board_image.empty())
		{
			cout << "Cannot open image file: " << filename << endl;
		}

		// Perform perspective transformation on current board.
		Mat current_board_pt = PerspectiveTransformation(current_board_image);

		// Find difference between empty board and current board (static background model).
		Mat difference;
		absdiff(current_board_pt, empty_board_pt, difference);
		Mat moving_points;
		cvtColor(difference, moving_points, COLOR_BGR2GRAY);
		threshold(moving_points, moving_points, 30, 255, THRESH_BINARY);
		moving_points = Opening(moving_points, GetStructuringElement3x3());
		moving_points = Dilate(moving_points, GetStructuringElement5x5());
		moving_points = Dilate(moving_points, GetStructuringElement5x5());
		//DisplayImage("moving", moving_points);

		// Get pieces using difference image as mask.
		Mat pieces_image = Mat::zeros(moving_points.size(), CV_8UC3);
		current_board_pt.copyTo(pieces_image, moving_points);
		//DisplayImage("Pieces", pieces_image);

		// Identify pieces in squares by observing the hue histogram in squares.
		int piece_count = 0;
		string squares_with_white_pieces = "";
		string squares_with_black_pieces = "";
		int square_number = 1;
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if (isBlackSquare(i * 50, j * 50))
				{
					int actual_square_contents = CheckBoardGroundTruth(square_number, GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);
					if (isPieceInSquare(moving_points, i * 50, j * 50))
					{
						if (isBlackPiece(pieces_image, i * 50, j * 50))
						{
							UpdateConfusionMatrix(confusion_matrix, BLACK_MAN_ON_SQUARE, actual_square_contents);
							squares_with_black_pieces += ((squares_with_black_pieces == "") ? "" : ",") + to_string(square_number);
						}
						else // it's a white piece
						{
							UpdateConfusionMatrix(confusion_matrix, WHITE_MAN_ON_SQUARE, actual_square_contents);
							squares_with_white_pieces += ((squares_with_white_pieces == "") ? "" : ",") + to_string(square_number);
						}
						piece_count++;
					}
					else // it's not a piece
					{
						UpdateConfusionMatrix(confusion_matrix, EMPTY_SQUARE, actual_square_contents);
					}
					square_number++;
				}
			}
		}
		//cout << "swwp: " << squares_with_white_pieces << endl;
		//cout << "swbp: " << squares_with_black_pieces << endl;
		//cout << "Squares with pieces: " << piece_count << endl;
	}

	

	// Could invert to get black squares as object pixels.
	//// Perform CCA on board image.
	//vector<vector<Point>> contours;
	//vector<Vec4i> hierarchy;
	//findContours(binary_empty_board, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	//Mat contours_image = Mat::zeros(binary_empty_board.size(), CV_8UC3);;
	//for (int contour = 0; (contour < contours.size()); contour++)
	//{
	//	Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
	//	drawContours(contours_image, contours, contour, colour, FILLED, 8, hierarchy);
	//}
	//DisplayImage("Empty Board Regions", contours_image);

	
}

void part3()
{
	cout << "part3" << endl;
}

void part4()
{
	cout << "part4" << endl;
}

void part5()
{
	cout << "part5" << endl;
}

// Print the given matrix (with an upper limit of elements to print).
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

// Display the given image with the provided name.
void DisplayImage(string name, Mat image)
{
    imshow(name, image);
}

// Get the number of object pixels in the given binary image.
int GetObjectPixelsInImage(Mat binary_image)
{
	int count = 0;
	for (int i = 0; i < binary_image.rows; i++)
	{
		for (int j = 0; j < binary_image.cols; j++)
		{
			int pixel = binary_image.at<uchar>(j, i);
			if (pixel == 255)
			{
				count++;
			}
		}
	}
	return count;
}


// Perform perspective transformation on board image.
Mat PerspectiveTransformation(Mat board_image)
{
	Mat result = Mat::zeros(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS, board_image.type());
	Point2f source[4] = { Point2f(114.0, 17.0), Point2f(53.0, 245.0), Point2f(355.0, 20.0), Point2f(433.0, 241.0) };
	Point2f destination[4] = { Point2f(0, 0), Point2f(0, BOARD_DIMENSIONS_IN_PIXELS), Point2f(BOARD_DIMENSIONS_IN_PIXELS, 0), Point2f(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS) };
	Mat perspective_matrix = getPerspectiveTransform(source, destination);
	warpPerspective(board_image, result, perspective_matrix, result.size());
	//DisplayImage("Perspective Transformation", result);
	return result;
}

// Create a 3x3 structuring element.
Mat GetStructuringElement3x3()
{
	return Mat();
}

// Create a 5x5 structuring element.
Mat GetStructuringElement5x5()
{
	Mat structuring_element(5, 5, CV_8U, Scalar(1));
	return structuring_element;
}

// Perform an erosion using given structuring element.
Mat Erode(Mat binary_image, Mat structuring_element)
{
	Mat eroded_image;
	erode(binary_image, eroded_image, structuring_element);
	return eroded_image;
}

// Perform a dliation using given structuring element.
Mat Dilate(Mat binary_image, Mat structuring_element)
{
	Mat dilated_image;
	dilate(binary_image, dilated_image, structuring_element);
	return dilated_image;
}

// Perform an opening using given structuring element.
Mat Opening(Mat binary_image, Mat structuring_element)
{
	Mat opened_image;
	morphologyEx(binary_image, opened_image, MORPH_OPEN, structuring_element);
	return opened_image;
}

// Perform a closing using given structuring element.
Mat Closing(Mat binary_image, Mat structuring_element)
{
	Mat closed_image;
	morphologyEx(binary_image, closed_image, MORPH_CLOSE, structuring_element);
	return closed_image;
}

// Check if a given square is a black square based on location.
bool isBlackSquare(int top_left_x, int top_left_y)
{
	bool is_black_square = false;
	if ((top_left_x % 100 == 0) && (top_left_y % 100 == 50)
		|| (top_left_x % 100 == 50) && (top_left_y % 100 == 0))
	{
		is_black_square = true;
	}
	return is_black_square;
}

// Get the number of object pixels found in a chess square.
int GetNumberOfObjectPixelsInSquare(Mat binary_image, int top_left_x, int top_left_y)
{
	int object_pixels = 0;
	for (int i = top_left_x; i < top_left_x + SQUARE_DIMENSIONS_IN_PIXELS; i++)
	{
		for (int j = top_left_y; j < top_left_y + SQUARE_DIMENSIONS_IN_PIXELS; j++)
		{
			int pixel = binary_image.at<uchar>(j, i);
			if (pixel == 255)
			{
				object_pixels++;
			}
		}
	}
	return object_pixels;
}

// Check if a given square contains a piece.
bool isPieceInSquare(Mat binary_image, int top_left_x, int top_left_y)
{
	bool is_piece_in_square = false;
	int object_pixels_in_square = GetNumberOfObjectPixelsInSquare(binary_image, top_left_x, top_left_y);
	if (object_pixels_in_square > PIXELS_IN_SQUARE / 4)
	{
		is_piece_in_square = true;
	}
	return is_piece_in_square;
}

// Check what colour piece is within a given square.
bool isBlackPiece(Mat rgb_image, int top_left_x, int top_left_y)
{
	bool is_black_piece = false;

	// Extract square from image.
	Range rows(top_left_x, top_left_x + SQUARE_DIMENSIONS_IN_PIXELS);
	Range cols(top_left_y, top_left_y + SQUARE_DIMENSIONS_IN_PIXELS);
	Mat square_image = rgb_image(cols, rows);
	//DisplayImage(to_string(top_left_y), square_image);

	// Histogram hue from square.
	Mat hist = HueHistogram(square_image, 25);
	//DisplayHistogram(to_string(top_left_y), hist, 25);
	//cout << "hist bin 0: " << to_string(hist.at<float>(0)) << endl; // Black Background
	//cout << "hist bin 1: " << to_string(hist.at<float>(1)) << endl; // Black piece
	//cout << "hist bin 2: " << to_string(hist.at<float>(2)) << endl; // White piece

	// Compare bins.
	if (hist.at<float>(1) > hist.at<float>(2))
	{
		is_black_piece = true;
	}

	return is_black_piece;
}

// Check the ground truth for what the specified square should contain.
int CheckBoardGroundTruth(int square_number, string white_pieces, string black_pieces)
{
	int ground_truth = EMPTY_SQUARE;

	// Check if it should be a white piece.
	regex rgx(",");
	sregex_token_iterator white_iterator(white_pieces.begin(), white_pieces.end(), rgx, -1);
	sregex_token_iterator white_end;
	for (; ground_truth == EMPTY_SQUARE && white_iterator != white_end; ++white_iterator)
	{
		string current_square = *white_iterator;
		if (current_square == to_string(square_number) || (current_square == "K" + to_string(square_number)))
		{
			ground_truth = WHITE_MAN_ON_SQUARE;
		}
	}

	// Check if it should be a black piece.
	sregex_token_iterator black_iterator(black_pieces.begin(), black_pieces.end(), rgx, -1);
	sregex_token_iterator black_end;
	for (; ground_truth == EMPTY_SQUARE && black_iterator != black_end; ++black_iterator)
	{
		string current_square = *black_iterator;
		if (current_square == to_string(square_number) || (current_square == "K" + to_string(square_number)))
		{
			ground_truth = BLACK_MAN_ON_SQUARE;
		}
	}

	return ground_truth;
}


// Update the confusion matrix based on what was detected in the square and what was recorded in the ground truth.
void UpdateConfusionMatrix(int confusion_matrix[3][3], int detected_square_contents, int actual_square_contents)
{
	switch (detected_square_contents)
	{
		case(EMPTY_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					confusion_matrix[0][0]++;
					break;
				case(WHITE_MAN_ON_SQUARE):
					confusion_matrix[0][1]++;
					cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[0][2]++;
					cout << "misclassification" << endl;
					break;
			}
			break;
		case(WHITE_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					confusion_matrix[1][0]++;
					cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					confusion_matrix[1][1]++;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[1][2]++;
					cout << "misclassification" << endl;
					break;
			}
			break;
		case(BLACK_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					confusion_matrix[2][0]++;
					cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					confusion_matrix[2][1]++;
					cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[2][2]++;
					break;
			}
			break;
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

// Display histogram.
void DisplayHistogram(string name, Mat hist, int bins)
{
    int w = 400, h = 400;
    int histSize = MAX(bins, 2);
    int bin_w = cvRound((double)w / histSize);
    Mat histImg = Mat::zeros(h, w, CV_8UC3);
    for (int i = 0; i < bins; i++)
    {
        rectangle(histImg, Point(i * bin_w, h), Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)), Scalar(0, 0, 255), FILLED);
    }
    imshow(name, histImg);
}

//// Backproject the given histogram onto the image.
//Mat Backproject(int, void*, Mat hist, Mat* image)
//{
//    // Backproject the histogram onto image.
//    Mat backproj;
//    float hue_range[] = { 0, 180 };
//    const float* ranges[] = { hue_range };
//    calcBackProject(image, 1, 0, hist, backproj, ranges, 1, true);
//    return backproj;
//}
//
//// Perform backprojection.
//void HistogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins) {
//    // Histogram hue in sample image.
//    Mat histogram = HueHistogram(sample_image, bins);
//
//    // Print histogram.
//    cout << name << " histogram:\n" << histogram << endl;
//
//    // Show histogram.
//    //DisplayHistogram(name, histogram, bins);
//
//    // Extract Hue channel from RGB image.
//    Mat image = ExtractHue(rgb_image);
//
//    // Backproject
//    Mat backprojection = Backproject(0, 0, histogram, &image);
//
//    // Print backprojection.
//    PrintMatrix(name + " backprojection", backprojection, 100);
//
//    // Show backprojection.
//    DisplayImage(name + " backprojection", backprojection);
//
//    // Threshold.
//    Mat thresholdedImage;
//    threshold(backprojection, thresholdedImage, 127, 255, THRESH_BINARY);
//
//    // Print thresholded image.
//    PrintMatrix(name + " thresholded", thresholdedImage, 100);
//
//    // Show thresholded image.
//    //DisplayImage(name + " thresholded", thresholdedImage);
//}

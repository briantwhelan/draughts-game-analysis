#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
#include <regex>
#include <string>
#include <map>
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

// Constant definitions.
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

// struct definitions.
// Struct to store info on differences in squares tracked across frames.
struct DifferenceLog
{
	// The frame number in which the first relevant difference was detected.
	int first_frame_number;
	// The frequency of the difference.
	int frequency;

	DifferenceLog(int frame_number)
	{
		this->first_frame_number = frame_number;
		this->frequency = 1;
	}
};

// Struct to store info on changes to square.
struct SquareChange
{
	// The square number that has changed.
	int square_number;
	// The frame number in which the square changed state.
	int frame_number;
	// The square state before the change.
	int before;
	// The square state after the change.
	int after;

	SquareChange(int square_number, int frame_number, int before, int after)
	{
		this->square_number = square_number;
		this->frame_number = frame_number;
		this->before = before;
		this->after = after;
	}
};

// Struct to store move info.
struct Move
{
	// Frame number of move.
	int frame_number;
	// The square number from which the piece moved.
	int from;
	// The square number to which the piece moved.
	int to;
	// The piece that has moved.
	int piece;

	Move(int frame_number, int from, int to, int piece)
	{
		this->frame_number = frame_number;
		this->from = from;
		this->to = to;
		this->piece = piece;
	}
};

// Function definitions.
void part1(Mat black_pieces_image, Mat white_pieces_image, Mat black_squares_image, Mat white_squares_image);
void part2(Mat empty_board_image, int confusion_matrix[3][3], Mat white_pieces_image, Mat black_pieces_image);
void part3(Mat empty_board_image, VideoCapture video);
void part4(Mat empty_board_image);
void part5(Mat empty_board_image, int extended_confusion_matrix[5][5]);

void printMatrix(string name, Mat matrix, int limit);
void displayImage(string name, Mat image);
int getObjectPixelsInImage(Mat binary_image);

Mat perspectiveTransformation(Mat board_image);
Mat getStructuringElement3x3();
Mat getStructuringElement5x5();

Mat erode(Mat binary_image, Mat structuring_element);
Mat dilate(Mat binary_image, Mat structuring_element);
Mat opening(Mat binary_image, Mat structuring_element);
Mat closing(Mat binary_image, Mat structuring_element);

int getSquare(int x, int y);
void getSquareCoordinates(int square_number, int coordinates[2]);
bool isBlackSquare(int top_left_x, int top_left_y);
int getNumberOfObjectPixelsInSquare(Mat binary_image, int top_left_x, int top_left_y);
bool isPieceInSquare(Mat binary_image, int top_left_x, int top_left_y);
bool isPieceInSquare(Mat binary_image, int square_number);
bool isBlackPiece(Mat rgb_image, int top_left_x, int top_left_y);
bool isBlackPiece(Mat rgb_image, int square_number);
bool isValidMove(int previous_board[NUMBER_OF_SQUARES], int current_board[NUMBER_OF_SQUARES], int from, int to);
void executeMove(int previous_board[NUMBER_OF_SQUARES], int current_board[NUMBER_OF_SQUARES], int from, int to);
bool isKing(Mat binary_image, int top_left_x, int top_left_y);
int checkBoardGroundTruth(int square_number, string white_pieces, string black_pieces);
int checkBoardGroundTruthWithKings(int square_number, string white_pieces, string black_pieces);
void updateConfusionMatrix(int confusion_matrix[3][3], int detected_square_contents, int actual_square_contents);
void updateExtendedConfusionMatrix(int extended_confusion_matrix[5][5], int detected_square_contents, int actual_square_contents);

Mat extractHue(Mat rgb_image);
Mat hueHistogram(Mat image, int bins);
void displayHistogram(string name, Mat hist, int bins);
Mat backproject(int, void*, Mat image, Mat* hue);
void histogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins);

void houghTransforms(Mat board_image);
void contourFollowing(Mat board_image);
void findCorners(Mat board_image);

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

// Data provided: Approx frame no, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const Move GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[] = {
	{17, 9, 13, WHITE_MAN_ON_SQUARE}, {37, 24, 20, BLACK_MAN_ON_SQUARE},		// 1. 9 - 13 24 - 20
	{50, 6, 9, WHITE_MAN_ON_SQUARE}, {65, 22, 17, BLACK_MAN_ON_SQUARE},			// 2. 6 - 9 22 - 17
	{85, 13, 22, WHITE_MAN_ON_SQUARE}, {108, 26, 17, BLACK_MAN_ON_SQUARE},		//	3. 13 - 22 26 - 17
	{123, 9, 13, WHITE_MAN_ON_SQUARE}, {161, 30, 26, BLACK_MAN_ON_SQUARE},		//	4. 9 - 13 30 - 26
	{180, 13, 22, WHITE_MAN_ON_SQUARE}, {201, 25, 18, BLACK_MAN_ON_SQUARE},		//	5. 13 - 22 25 - 18
	{226, 12, 16, WHITE_MAN_ON_SQUARE}, {244, 18, 14, BLACK_MAN_ON_SQUARE},		//	6. 12 - 16 18 - 14
	{266, 10, 17, WHITE_MAN_ON_SQUARE}, {285, 21, 14, BLACK_MAN_ON_SQUARE},		//	7. 10 - 17 21 - 14
	{308, 2, 6, WHITE_MAN_ON_SQUARE}, {326, 26, 22, BLACK_MAN_ON_SQUARE},		//	8. 2 - 6 26 - 22
	{343, 6, 9, WHITE_MAN_ON_SQUARE}, {362, 22, 18, BLACK_MAN_ON_SQUARE},		//	9. 6 - 9 22 - 18
	{393, 11, 15, WHITE_MAN_ON_SQUARE}, {420, 18, 2, BLACK_MAN_ON_SQUARE},		//	10. 11 - 15 18 - 2
	{453, 9, 18, WHITE_MAN_ON_SQUARE}, {472, 23, 14, BLACK_MAN_ON_SQUARE},		//	11. 9 - 18 23 - 14
	{506, 3, 7, WHITE_MAN_ON_SQUARE}, {530, 20, 11, BLACK_MAN_ON_SQUARE},		//	12. 3 - 7 20 - 11
	{546, 7, 16, WHITE_MAN_ON_SQUARE}, {582, 2, 7, BLACK_MAN_ON_SQUARE},		//	13. 7 - 16 2 - 7
	{617, 8, 11, WHITE_MAN_ON_SQUARE}, {641, 27, 24, BLACK_MAN_ON_SQUARE},		//	14. 8 - 11 27 - 24
	{673, 1, 6, WHITE_MAN_ON_SQUARE}, {697, 7, 2, BLACK_MAN_ON_SQUARE},			//	15. 1 - 6 7 - 2
	{714, 6, 9, WHITE_MAN_ON_SQUARE}, {728, 14, 10, BLACK_MAN_ON_SQUARE},		//	16. 6 - 9 14 - 10
	{748, 9, 14, WHITE_MAN_ON_SQUARE}, {767, 10, 7, BLACK_MAN_ON_SQUARE},		//	17. 9 - 14 10 - 7
	{781, 14, 17, WHITE_MAN_ON_SQUARE}, {801, 7, 3, BLACK_MAN_ON_SQUARE},		//	18. 14 - 17 7 - 3
	{814, 11, 15, WHITE_MAN_ON_SQUARE}, {859, 24, 20, BLACK_MAN_ON_SQUARE},		//	19. 11 - 15 24 - 20
	{870, 16, 19, WHITE_MAN_ON_SQUARE}, {891, 3, 7, BLACK_MAN_ON_SQUARE},		//	20. 16 - 19 3 - 7
	{923, 15, 18, WHITE_MAN_ON_SQUARE}, {936, 7, 10, BLACK_MAN_ON_SQUARE},		//	21. 15 - 18 7 - 10
	{955, 18, 22, WHITE_MAN_ON_SQUARE}, {995, 10, 14, BLACK_MAN_ON_SQUARE},		//	22. 18 - 22 10 - 14
	{1014, 17, 21, WHITE_MAN_ON_SQUARE}, {1034, 14, 17, BLACK_MAN_ON_SQUARE},	//	23. 17 - 21 14 - 17
	{1058, 21, 25, WHITE_MAN_ON_SQUARE}, {1075, 17, 26, BLACK_MAN_ON_SQUARE},	//	24. 21 - 25 17 - 26
	{1104, 25, 30, WHITE_MAN_ON_SQUARE}, {1129, 31, 27, BLACK_MAN_ON_SQUARE},	//	25. 25 - 30 31 - 27
	{1147, 30, 23, WHITE_MAN_ON_SQUARE}, {1166, 27, 18, BLACK_MAN_ON_SQUARE},	//	26. 30 - 23 27 - 18
	{1182, 19, 23, WHITE_MAN_ON_SQUARE}, {1201, 18, 15, BLACK_MAN_ON_SQUARE},	//	27. 19 - 23 18 - 15
	{1213, 23, 26, WHITE_MAN_ON_SQUARE}, {1243, 15, 11, BLACK_MAN_ON_SQUARE},	//	28. 23 - 26 15 - 11
	{1266, 26, 31, WHITE_MAN_ON_SQUARE}, {1280, 32, 27, BLACK_MAN_ON_SQUARE},	//	29. 26 - 31 32 - 27
	{1298, 31, 24, WHITE_MAN_ON_SQUARE}, {1324, 28, 19, BLACK_MAN_ON_SQUARE},	//	30. 31 - 24 28 - 19
	{1337, 5, 9, WHITE_MAN_ON_SQUARE}, {1358, 29, 25, BLACK_MAN_ON_SQUARE},		//	31. 5 - 9 29 - 25
	{1387, 9, 14, WHITE_MAN_ON_SQUARE}, {1410, 25, 22, BLACK_MAN_ON_SQUARE},	//	32. 9 - 14 25 - 22
	{1438, 14, 18, WHITE_MAN_ON_SQUARE}, {1450, 22, 15, BLACK_MAN_ON_SQUARE},	//	33. 14 - 18 22 - 15
	{1465, 4, 8, WHITE_MAN_ON_SQUARE}, {1490, 11, 4, BLACK_MAN_ON_SQUARE} 		//	34. 4 - 8 11 - 4
};

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
		/* RECOMMENDED TO COMMENT OUT PARTS AND RUN EACH INDIVIDUALLY*/
		// Classify the pixels.
		part1(black_pieces_image, white_pieces_image, black_squares_image, white_squares_image);

		// Compute confusion matrix for pieces in squares.
		int confusion_matrix[3][3] = {{0, 0, 0},
									   {0, 0, 0}, 
									   {0, 0, 0} };
		part2(static_background_image, confusion_matrix, white_pieces_image, black_pieces_image);
		cout << "Confusion Matrix:\n"
			<< "\tGT_NP\tGT_WP\tGT_BP\n"
			<< "D_NP\t" << confusion_matrix[0][0] << "\t" << confusion_matrix[0][1] << "\t" << confusion_matrix[0][2] << "\n" 
			<< "D_WP\t" << confusion_matrix[1][0] << "\t" << confusion_matrix[1][1] << "\t" << confusion_matrix[1][2] << "\n" 
			<< "D_BP\t" << confusion_matrix[2][0] << "\t" << confusion_matrix[2][1] << "\t" << confusion_matrix[2][2] << endl;

		// Record moves in video.
		part3(static_background_image, video);

		// Identify four corners of the chessboard.
		part4(static_background_image);

		// Distinguish between normal pieces and kings.
		int extended_confusion_matrix[5][5] = {{0, 0, 0, 0, 0},
									            {0, 0, 0, 0, 0},
									            {0, 0, 0, 0, 0} };
		part5(static_background_image, extended_confusion_matrix);
		cout << "Confusion Matrix:\n"
			<< "\tGT_NP\tGT_WM\tGT_WK\tGT_BM\tGT_BK\n"
			<< "D_NP\t" << extended_confusion_matrix[0][0] << "\t" << extended_confusion_matrix[0][1] << "\t" << extended_confusion_matrix[0][2] << "\t" << extended_confusion_matrix[0][3] << "\t" << extended_confusion_matrix[0][4] << "\n"
			<< "D_WM\t" << extended_confusion_matrix[1][0] << "\t" << extended_confusion_matrix[1][1] << "\t" << extended_confusion_matrix[1][2] << "\t" << extended_confusion_matrix[1][3] << "\t" << extended_confusion_matrix[1][4] << "\n"
			<< "D_WK\t" << extended_confusion_matrix[1][0] << "\t" << extended_confusion_matrix[2][1] << "\t" << extended_confusion_matrix[2][2] << "\t" << extended_confusion_matrix[2][3] << "\t" << extended_confusion_matrix[2][4] << "\n"
			<< "D_BM\t" << extended_confusion_matrix[1][0] << "\t" << extended_confusion_matrix[3][1] << "\t" << extended_confusion_matrix[3][2] << "\t" << extended_confusion_matrix[3][3] << "\t" << extended_confusion_matrix[3][4] << "\n"
			<< "D_BK\t" << extended_confusion_matrix[2][0] << "\t" << extended_confusion_matrix[4][1] << "\t" << extended_confusion_matrix[4][2] << "\t" << extended_confusion_matrix[4][3] << "\t" << extended_confusion_matrix[4][4] << endl;
	}
}

void part1(Mat black_pieces_image, Mat white_pieces_image, Mat black_squares_image, Mat white_squares_image)
{
	// Load image to classify.
	string source_filename = "Media/DraughtsGame1Move0.JPG";
	Mat source_image = imread(source_filename, -1);
	if (source_image.empty())
	{
		cout << "Cannot open image file: " << source_filename << endl;
	}

	// Show source image.
	displayImage("Source", source_image);

	/*// Use k-means clustering.
	Mat data;
	source_image.convertTo(data, CV_32F);
	data = data.reshape(1, data.total());
	Mat labels, centers;
	kmeans(data, 4, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.5), 3, KMEANS_PP_CENTERS, centers);
	centers = centers.reshape(3, centers.rows);
	data = data.reshape(3, data.rows);
	Vec3f* p = data.ptr<Vec3f>();
	for (int i = 0; i < data.rows; i++) 
	{
		int center_id = labels.at<int>(i);
		p[i] = centers.at<Vec3f>(center_id);
	}
	source_image = data.reshape(3, source_image.rows);
	source_image.convertTo(source_image, CV_8U);
	displayImage("k-means", source_image);*/

	/*// Backprojections using hue histograms.
	int bins = 25;
	histogramAndBackproject("white pieces", white_pieces_image, source_image, bins);
	histogramAndBackproject("black pieces", black_pieces_image, source_image, bins);
	histogramAndBackproject("white squares", white_squares_image, source_image, bins);
	histogramAndBackproject("black squares", black_squares_image, source_image, bins);*/

	// Backproject piece and square samples onto source.
	Mat white_pieces_probs = BackProjection(source_image, white_pieces_image);
	Mat black_pieces_probs = BackProjection(source_image, black_pieces_image);
	Mat white_squares_probs = BackProjection(source_image, white_squares_image);
	Mat black_squares_probs = BackProjection(source_image, black_squares_image);
	Mat pieces_probs = JoinImagesHorizontally(white_pieces_probs, "white pieces", black_pieces_probs, "black pieces", 4, 255);
	Mat squares_probs = JoinImagesHorizontally(white_squares_probs, "white squares", black_squares_probs, "black squares", 4, 255);
	Mat backprojections = JoinImagesVertically(pieces_probs, "", squares_probs, "", 4, 255);
	displayImage("Backprojections", backprojections);

	// Compare probability images to classify image.
	const Vec3b WHITE = { 255, 255, 255 };
	const Vec3b BLACK = { 0, 0, 0 };
	const Vec3b RED = { 0, 0, 255 };
	const Vec3b GREEN = { 0, 255, 0};
	const Vec3b BLUE = { 255, 0, 0 };
	Mat classification_image = Mat::zeros(source_image.size(), CV_8UC3);
	for (int i = 0; i < classification_image.rows; i++)
	{
		for (int j = 0; j < classification_image.cols; j++)
		{
			int black_pieces_prob = black_pieces_probs.at<uchar>(i, j);
			int white_pieces_prob = white_pieces_probs.at<uchar>(i, j);
			int black_squares_prob = black_squares_probs.at<uchar>(i, j);
			int white_squares_prob = white_squares_probs.at<uchar>(i, j);
			int max_prob = max(max(max(black_pieces_prob, white_pieces_prob), black_squares_prob), white_squares_prob);
			if (max_prob > 127)
			{
				if (max_prob == white_pieces_prob)
				{
					classification_image.at<Vec3b>(i, j) = WHITE;
				}
				else if (max_prob == black_pieces_prob)
				{
					classification_image.at<Vec3b>(i, j) = BLACK;
				}
				else if (max_prob == white_squares_prob)
				{
					classification_image.at<Vec3b>(i, j) = RED;
				}
				else if (max_prob == black_squares_prob)
				{
					classification_image.at<Vec3b>(i, j) = GREEN;
				}
			}
			else // it's background
			{
				classification_image.at<Vec3b>(i, j) = BLUE;
			}
		}
	}
	//displayImage("Classification Image", classification_image);

	// Load ground truth image.
	string ground_truth_filename = "Media/DraughtsGame1Move0GroundTruth.png";
	Mat ground_truth_image = imread(ground_truth_filename, -1);
	if (ground_truth_image.empty())
	{
		cout << "Cannot open image file: " << ground_truth_filename << endl;
	}
	cvtColor(ground_truth_image, ground_truth_image, COLOR_BGRA2BGR);

	// Compare qualitatively with ground truth.
	Mat difference_image;
	absdiff(classification_image, ground_truth_image, difference_image);
	//displayImage("Difference with Ground Truth", difference_image);
	Mat binary_difference_image;
	cvtColor(difference_image, binary_difference_image, COLOR_BGR2GRAY);
	threshold(binary_difference_image, binary_difference_image, 30, 255, THRESH_BINARY);
	//displayImage("Binary Difference with Ground Truth", binary_difference_image);

	// Display results.
	Mat classification_images = JoinImagesHorizontally(ground_truth_image, "Ground Truth", classification_image, "Classification", 4, -1);
	cvtColor(binary_difference_image, binary_difference_image, COLOR_GRAY2BGR);
	Mat difference_images = JoinImagesHorizontally(difference_image, "Difference with Ground Truth", binary_difference_image, "Binary Difference with Ground Truth", 4, -1);
	Mat results = JoinImagesVertically(classification_images, "", difference_images, "", 4, 255);
	displayImage("Classifying the pixels", results);

	// Compare quantitatively with ground truth.
	int misclassifications = 0;
	int white_piece_misclassifications = 0;
	int black_piece_misclassifications = 0;
	int white_square_misclassifications = 0;
	int black_square_misclassifications = 0;
	int background_misclassifications = 0;
	for (int i = 0; i < classification_image.rows; i++)
	{
		for (int j = 0; j < classification_image.cols; j++)
		{
			Vec3b actual = ground_truth_image.at<Vec3b>(i, j);
			Vec3b predicted = classification_image.at<Vec3b>(i, j); 
			if (actual != predicted)
			{
				if (actual == WHITE)
				{
					white_piece_misclassifications++;
				}
				else if (actual == BLACK)
				{
					black_piece_misclassifications++;
				}
				else if (actual == RED)
				{
					white_square_misclassifications++;
				}
				else if (actual == GREEN)
				{
					black_square_misclassifications++;
				}
				else if (actual == BLUE)
				{
					background_misclassifications++;
				}
				misclassifications++;
			}
		}
	}
	cout << "Misclassifications\n" 
		<< "\tWhite pieces: " << white_piece_misclassifications
		<< "\n\tBlack pieces: " << black_piece_misclassifications
		<< "\n\tWhite squares: " << white_square_misclassifications
		<< "\n\tBlack squares: " << black_square_misclassifications 
		<< "\n\tBackground: " << background_misclassifications
		<< "\nTotal: " << misclassifications << endl;
}

void part2(Mat empty_board_image, int confusion_matrix[3][3], Mat white_pieces_image, Mat black_pieces_image)
{
	// Perform perspective transformation on empty board.
	Mat empty_board_pt = perspectiveTransformation(empty_board_image);
	
	// Otsu Threshold empty board image.
	Mat grey_board_image;
	cvtColor(empty_board_pt, grey_board_image, COLOR_BGR2GRAY);
	Mat thresholded_board_image;
	threshold(grey_board_image, thresholded_board_image, 127, 255, THRESH_BINARY | THRESH_OTSU);
	//displayImage("Thresholded Board Image", thresholded_board_image);

	// Tidy with opening to get binary empty board.
	Mat binary_empty_board = opening(thresholded_board_image, getStructuringElement5x5());
	//displayImage("Binary Empty Board", binary_empty_board);

	// Process all static images.
	for (int image_index = 0; image_index < NUMBER_OF_STATIC_IMAGES; image_index++)
	{
		// Load current board image.
		string filename = "Media/" + GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0];
		Mat current_board_image = imread(filename, -1);
		if (current_board_image.empty())
		{
			cout << "Cannot open image file: " << filename << endl;
		}
		//displayImage("Board", current_board_image);

		// Perform perspective transformation on current board.
		Mat current_board_pt = perspectiveTransformation(current_board_image);
		//displayImage("Board Perspective Transformation",current_board_pt);

		// Find difference between empty board and current board (static background model).
		Mat difference;
		absdiff(current_board_pt, empty_board_pt, difference);
		cvtColor(difference, difference, COLOR_BGR2GRAY);
		threshold(difference, difference, 30, 255, THRESH_BINARY);
		//displayImage("Difference", difference);
		Mat difference_transformed;
		difference_transformed = opening(difference, getStructuringElement3x3());
		difference_transformed = dilate(difference_transformed, getStructuringElement5x5());
		difference_transformed = dilate(difference_transformed, getStructuringElement5x5());
		//displayImage("Difference with Geometric Operations", difference_transformed);

		// Get pieces using difference image as mask.
		Mat pieces_image = Mat::zeros(difference.size(), CV_8UC3);
		current_board_pt.copyTo(pieces_image, difference_transformed);
		//displayImage("Pieces", pieces_image);

		// Display results.
		/*Mat display_image1 = difference.clone();
		cvtColor(display_image1, display_image1, COLOR_GRAY2BGR);
		Mat display_image2 = difference_transformed.clone();
		cvtColor(display_image2, display_image2, COLOR_GRAY2BGR);
		Mat img1 = JoinImagesHorizontally(current_board_pt, "Board Perspective Transformation", display_image1, "Difference", 4, -1);
		Mat img2 = JoinImagesHorizontally(display_image2, "Difference with Geometric Operations", pieces_image, "Pieces", 4, -1);
		Mat results = JoinImagesVertically(img1, "", img2, "", 4, 255);
		displayImage("Identifying the squares and pieces", results);*/		

		// Identify pieces in squares.
		int square_number = 1;
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if (isBlackSquare(i * 50, j * 50))
				{
					int actual_square_contents = checkBoardGroundTruth(square_number, GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);
					if (isPieceInSquare(difference_transformed, i * 50, j * 50))
					{
						if (isBlackPiece(pieces_image, i * 50, j * 50))
						{
							updateConfusionMatrix(confusion_matrix, BLACK_MAN_ON_SQUARE, actual_square_contents);
						}
						else // it's a white piece
						{
							updateConfusionMatrix(confusion_matrix, WHITE_MAN_ON_SQUARE, actual_square_contents);
						}
					}
					else // it's empty
					{
						updateConfusionMatrix(confusion_matrix, EMPTY_SQUARE, actual_square_contents);
					}
					square_number++;
				}
			}
		}
	}
}

void part3(Mat empty_board_image, VideoCapture video)
{
	// Perform perspective transformation on empty board.
	Mat empty_board_pt = perspectiveTransformation(empty_board_image);

	// Otsu Threshold empty board image.
	Mat grey_board_image;
	cvtColor(empty_board_pt, grey_board_image, COLOR_BGR2GRAY);
	Mat thresholded_board_image;
	threshold(grey_board_image, thresholded_board_image, 127, 255, THRESH_BINARY | THRESH_OTSU);
	//DisplayImage("Thresholded Board Image", thresholded_board_image);

	// Tidy with opening to get binary empty board.
	Mat binary_empty_board = opening(thresholded_board_image, getStructuringElement5x5());
	//DisplayImage("Binary Empty Board", binary_empty_board);
	
	// Keep track of board state.
	int piece_count = 24;
	int previous_board[NUMBER_OF_SQUARES] = {
		WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE,
		WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE,
		WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE, WHITE_MAN_ON_SQUARE,
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE,
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE,
		BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE,
		BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE,
		BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE, BLACK_MAN_ON_SQUARE
	};
	int current_board[NUMBER_OF_SQUARES] = { 
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, 
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE,
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, 
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE,
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, 
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE,
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, 
		EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE, EMPTY_SQUARE 
	};

	// Process video frame by frame.
	Mat current_frame;
	video.set(cv::CAP_PROP_POS_FRAMES, 1);
	video >> current_frame;
	double last_time = static_cast<double>(getTickCount());
	double frame_rate = video.get(cv::CAP_PROP_FPS);
	double time_between_frames = 1000.0 / frame_rate;
	map<int, DifferenceLog*> square_diff_log;
	vector<SquareChange*> square_changes;
	vector<Move*> moves;
	for(int frame = 0; !current_frame.empty(); frame++)
	{
		// Perform perspective transformation on current board.
		Mat current_board_pt = perspectiveTransformation(current_frame);

		// Find difference between empty board and current board (static background model).
		Mat difference;
		absdiff(current_board_pt, empty_board_pt, difference);
		Mat moving_points;
		cvtColor(difference, moving_points, COLOR_BGR2GRAY);
		threshold(moving_points, moving_points, 30, 255, THRESH_BINARY);
		moving_points = opening(moving_points, getStructuringElement3x3());
		moving_points = dilate(moving_points, getStructuringElement5x5());
		moving_points = dilate(moving_points, getStructuringElement5x5());
		//displayImage("moving", moving_points);
		int object_pixels = getObjectPixelsInImage(moving_points);
		//cout << "Frame " << frame << " object Pixels: " << object_pixels << endl;

		// Only consider frames with certain number of object pixels.
		if (object_pixels < (piece_count * PIXELS_IN_SQUARE))
		{
			cout << "Frame " << frame << endl;
			// Get pieces using difference image as mask.
			Mat pieces_image = Mat::zeros(moving_points.size(), CV_8UC3);
			current_board_pt.copyTo(pieces_image, moving_points);
			//displayImage("Pieces", pieces_image);

			// Detect state of current board.
			int detected_piece_count = 0;
			for (int square_number = 1; square_number <= NUMBER_OF_SQUARES; square_number++)
			{
				if (isPieceInSquare(moving_points, square_number))
				{
					if (isBlackPiece(pieces_image, square_number))
					{
						current_board[square_number - 1] = BLACK_MAN_ON_SQUARE;
					}
					else // it's a white piece
					{
						current_board[square_number - 1] = WHITE_MAN_ON_SQUARE;
					}
					detected_piece_count++;
				}
				else // it's not a piece
				{
					current_board[square_number - 1] = EMPTY_SQUARE;
				}
			}

			// Record differences between frames.
			vector<int> diffs;
			for (int square_number = 0; square_number < NUMBER_OF_SQUARES; square_number++)
			{
				// Difference must go from piece to empty or vice versa.
				if (previous_board[square_number] != current_board[square_number]
					&& ((previous_board[square_number] == EMPTY_SQUARE && !(current_board[square_number] == EMPTY_SQUARE))
						|| (!(previous_board[square_number] == EMPTY_SQUARE) && current_board[square_number] == EMPTY_SQUARE)))
				{
					diffs.push_back(square_number);
					//cout << "Difference at " << square_number + 1 << "\t Was: " << previous_board[square_number] << "\tNow: " << current_board[square_number] << endl;
				}
			}

			// Check if difference persists across previous frames.
			for (int i = 0; i < diffs.size(); i++)
			{
				int square_number = diffs[i];
				if (square_diff_log.count(square_number) == 1)
				{
					map<int, DifferenceLog*>::iterator it = square_diff_log.find(square_number);
					DifferenceLog* log = it->second;
					// Update square if difference persists across 5 of the previous 10 frames.
					if (frame - log->first_frame_number <= 10)
					{
						if (log->frequency + 1 == 5)
						{
							
							cout << "\tUpdate square " << square_number + 1 << " from " << previous_board[square_number] << " to " << current_board[square_number] << endl;
							SquareChange* change = new SquareChange(square_number, frame, previous_board[square_number], current_board[square_number]);
							square_changes.push_back(change);
							previous_board[square_number] = current_board[square_number];
							square_diff_log.erase(square_number);
						}
						else
						{
							log->frequency++;
						}
					}
					else
					{
						square_diff_log.erase(square_number);
						square_diff_log.insert({ square_number, new DifferenceLog(frame) });
					}

				}
				else
				{
					square_diff_log.insert({ square_number, new DifferenceLog(frame) });
				}
			}

			// Identify moves from updated squares.
			for (int i = square_changes.size() - 1; i >= 0; i--)
			{
				// Only consider changes in last 10 frames.
				if (abs(frame - square_changes[i]->frame_number) > 10)
				{
					break;
				}
				SquareChange* square_change1 = square_changes[i];
				for (int j = square_changes.size() - 1; j >= 0; j--)
				{
					// Only consider changes in last 10 frames.
					if (abs(frame - square_changes[j]->frame_number) > 10)
					{
						break;
					}
					SquareChange* square_change2 = square_changes[j];
					// Check for moves.
					if ((abs(square_change1->frame_number - square_change2->frame_number) <= 10)
						&& (square_change1->before != EMPTY_SQUARE) && (square_change2->before == EMPTY_SQUARE)
						//&& (square_change1->before == square_change2->after) 
						&& square_change1->square_number != square_change2->square_number)
					{
						int from = square_change1->square_number + 1;
						int to = square_change2->square_number + 1;

						// Check if move already recorded.
						bool moveRecorded = false;
						for (int k = 0; !moveRecorded && k < moves.size(); k++)
						{
							Move* recorded_move = moves[k];
							if (abs(frame - recorded_move->frame_number) <= 10
								&& recorded_move->from == from
								&& recorded_move->to == to
								&& recorded_move->piece == square_change1->before)
							{
								moveRecorded = true;
							}
						}
						// Record move.
						if (!moveRecorded)
						{
							cout << "\t Move from " << square_change1->square_number + 1 << " to " << square_change2->square_number + 1 << endl;
							Move* move = new Move(frame, from, to, square_change1->before);
							moves.push_back(move);
						}
					}
				}
			}				

			//// Check for any valid moves.
			//bool moveMade = false;
			//for (int i = 0; !moveMade && i < square_changes.size(); i++)
			//{
			//	SquareChange* square_change1 = square_changes[i];
			//	for (int j = 0; !moveMade && j < square_changes.size(); j++)
			//	{
			//		SquareChange* square_change2 = square_changes[j];
			//		int from = square_change1->square_number + 1;
			//		int to = square_change2->square_number + 1;
			//		if (i != j && !moveMade && isValidMove(previous_board, current_board, from, to))
			//		{
			//			cout << "Frame " << frame << endl;
			//			cout << "\tSwap from " << from << " to " << to << endl;
			//			executeMove(previous_board, current_board, from, to);
			//			moveMade = true;
			//		}
			//	}
			//}

			piece_count = detected_piece_count;
		}
		imshow("Draughts video", current_board_pt);
		double current_time = static_cast<double>(getTickCount());
		double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
		int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
		last_time = current_time;
		video >> current_frame;
		char c = cv::waitKey(1);  // If you replace delay with 1 it will play the video as quickly as possible.
	}
	cv::destroyAllWindows();

	// Compare moves with ground truth.
	cout << moves.size() << endl;
	int missed_moves = 0;
	for (const Move actual_move : GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES)
	{
		bool moveDetected = false;
		for (Move* detected_move : moves)
		{
			if (abs(actual_move.frame_number - detected_move->frame_number) <= 10	
				//&& actual_move.piece == detected_move->piece
				&& actual_move.from == detected_move->from
				&& actual_move.to == detected_move->to)
			{
				cout << "Move detected - Frame:" << detected_move->frame_number 
					<< "\tFrom:" << detected_move->from 
					<< "\tTo:" << detected_move->to << endl;
				moveDetected = true;
				break;
			}
		}
		if(!moveDetected)
		{
			cout << "\tMove missed - Frame:" << actual_move.frame_number
				<< "\tFrom:" << actual_move.from
				<< "\tTo:" << actual_move.to << endl;
			missed_moves++;
		}
	}
	cout << "Missed " << missed_moves << " moves." << endl;
}

void part4(Mat board_image)
{
	// Use of the Hough transformation for lines spanning the complete image.
	houghTransforms(board_image);
	
	// Use of contour following and straight line segment extraction.
	contourFollowing(board_image);
	
	// Use of the findChessboardCorners() routine in OpenCV.
	findCorners(board_image);
}

void part5(Mat empty_board_image, int extended_confusion_matrix[5][5])
{
	// Perform perspective transformation on empty board.
	Mat empty_board_pt = perspectiveTransformation(empty_board_image);

	// Otsu Threshold empty board image.
	Mat grey_board_image;
	cvtColor(empty_board_pt, grey_board_image, COLOR_BGR2GRAY);
	Mat thresholded_board_image;
	threshold(grey_board_image, thresholded_board_image, 127, 255, THRESH_BINARY | THRESH_OTSU);
	//displayImage("Thresholded Board Image", thresholded_board_image);

	// Tidy with opening to get binary empty board.
	Mat binary_empty_board = opening(thresholded_board_image, getStructuringElement5x5());
	//displayImage("Binary Empty Board", binary_empty_board);

	// Process all static images.
	for (int image_index = 0; image_index < NUMBER_OF_STATIC_IMAGES; image_index++)
	{
		//cout << "Image " << image_index << endl;
		// Load current board image.
		string filename = "Media/" + GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0];
		Mat current_board_image = imread(filename, -1);
		if (current_board_image.empty())
		{
			cout << "Cannot open image file: " << filename << endl;
		}

		// Perform perspective transformation on current board.
		Mat current_board_pt = perspectiveTransformation(current_board_image);

		// Find difference between empty board and current board (static background model).
		Mat difference;
		absdiff(current_board_pt, empty_board_pt, difference);
		Mat moving_points;
		cvtColor(difference, moving_points, COLOR_BGR2GRAY);
		threshold(moving_points, moving_points, 30, 255, THRESH_BINARY);
		moving_points = opening(moving_points, getStructuringElement3x3());
		moving_points = dilate(moving_points, getStructuringElement5x5());
		moving_points = dilate(moving_points, getStructuringElement5x5());
		moving_points = closing(moving_points, getStructuringElement3x3());
		//displayImage("moving", moving_points);

		// Get pieces using difference image as mask.
		Mat pieces_image = Mat::zeros(moving_points.size(), CV_8UC3);
		current_board_pt.copyTo(pieces_image, moving_points);
		//displayImage("Pieces", pieces_image);

		// Identify pieces in squares by observing the hue histogram in squares.
		int square_number = 1;
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if (isBlackSquare(i * 50, j * 50))
				{
					//cout << square_number << endl;
					int actual_square_contents = checkBoardGroundTruthWithKings(square_number, GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);
					if (isPieceInSquare(moving_points, i * 50, j * 50))
					{
						if (isBlackPiece(pieces_image, i * 50, j * 50))
						{
							if (isKing(moving_points, i * 50, j * 50))
							{
								updateExtendedConfusionMatrix(extended_confusion_matrix, BLACK_KING_ON_SQUARE, actual_square_contents);
							}
							else // it's a black man
							{
								updateExtendedConfusionMatrix(extended_confusion_matrix, BLACK_MAN_ON_SQUARE, actual_square_contents);
							}
						}
						else // it's a white piece
						{
							if (isKing(moving_points, i * 50, j * 50))
							{
								updateExtendedConfusionMatrix(extended_confusion_matrix, WHITE_KING_ON_SQUARE, actual_square_contents);
							}
							else // it's a white man
							{
								updateExtendedConfusionMatrix(extended_confusion_matrix, WHITE_MAN_ON_SQUARE, actual_square_contents);
							}
						}
					}
					else // it's not a piece
					{
						updateExtendedConfusionMatrix(extended_confusion_matrix, EMPTY_SQUARE, actual_square_contents);
					}
					square_number++;
				}
			}
		}
	}
}

// Print the given matrix (with an upper limit of elements to print).
void printMatrix(string name, Mat matrix, int limit) 
{
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
void displayImage(string name, Mat image)
{
    imshow(name, image);
}

// Get the number of object pixels in the given binary image.
int getObjectPixelsInImage(Mat binary_image)
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
Mat perspectiveTransformation(Mat board_image)
{
	Mat result = Mat::zeros(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS, board_image.type());
	Point2f source[4] = { Point2f(114.0, 17.0), Point2f(53.0, 245.0), Point2f(355.0, 20.0), Point2f(433.0, 241.0) };
	Point2f destination[4] = { Point2f(0, 0), Point2f(0, BOARD_DIMENSIONS_IN_PIXELS), Point2f(BOARD_DIMENSIONS_IN_PIXELS, 0), Point2f(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS) };
	Mat perspective_matrix = getPerspectiveTransform(source, destination);
	warpPerspective(board_image, result, perspective_matrix, result.size());
	//displayImage("Perspective Transformation", result);
	return result;
}

// Create a 3x3 structuring element.
Mat getStructuringElement3x3()
{
	return Mat();
}

// Create a 5x5 structuring element.
Mat getStructuringElement5x5()
{
	Mat structuring_element(5, 5, CV_8U, Scalar(1));
	return structuring_element;
}

// Perform an erosion using given structuring element.
Mat erode(Mat binary_image, Mat structuring_element)
{
	Mat eroded_image;
	erode(binary_image, eroded_image, structuring_element);
	return eroded_image;
}

// Perform a dliation using given structuring element.
Mat dilate(Mat binary_image, Mat structuring_element)
{
	Mat dilated_image;
	dilate(binary_image, dilated_image, structuring_element);
	return dilated_image;
}

// Perform an opening using given structuring element.
Mat opening(Mat binary_image, Mat structuring_element)
{
	Mat opened_image;
	morphologyEx(binary_image, opened_image, MORPH_OPEN, structuring_element);
	return opened_image;
}

// Perform a closing using given structuring element.
Mat closing(Mat binary_image, Mat structuring_element)
{
	Mat closed_image;
	morphologyEx(binary_image, closed_image, MORPH_CLOSE, structuring_element);
	return closed_image;
}

// Get square number from top-left coordinates.
int getSquare(int x, int y)
{
	int square_number = -1;
	int column = (x / SQUARE_DIMENSIONS_IN_PIXELS);
	int start = (NUMBER_OF_SQUARES_ON_EACH_SIDE / 2) * column;
	int offset = (y / (2 * SQUARE_DIMENSIONS_IN_PIXELS)) + 1;
	square_number = start + offset;
	return square_number;
}

// Get square top-left coordinates from square number.
void getSquareCoordinates(int square_number, int coordinates[2])
{
	if ((square_number - 1) % 8 < 4)
	{
		coordinates[0] = (square_number - 1)/NUMBER_OF_SQUARES_ON_EACH_SIDE * (2 * SQUARE_DIMENSIONS_IN_PIXELS);
		coordinates[1] = SQUARE_DIMENSIONS_IN_PIXELS + ((square_number - 1) % (NUMBER_OF_SQUARES_ON_EACH_SIDE/2)) * (2 * SQUARE_DIMENSIONS_IN_PIXELS);
	}
	else // ((square_number - 1) % 8 >= 4)
	{
		coordinates[0] = SQUARE_DIMENSIONS_IN_PIXELS + ((square_number - 1)/NUMBER_OF_SQUARES_ON_EACH_SIDE) * (2 * SQUARE_DIMENSIONS_IN_PIXELS);
		coordinates[1] = ((square_number - 1) % (NUMBER_OF_SQUARES_ON_EACH_SIDE/2)) * (2 * SQUARE_DIMENSIONS_IN_PIXELS);
	}
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
int getNumberOfObjectPixelsInSquare(Mat binary_image, int top_left_x, int top_left_y)
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
	int object_pixels_in_square = getNumberOfObjectPixelsInSquare(binary_image, top_left_x, top_left_y);
	if (object_pixels_in_square > PIXELS_IN_SQUARE / 4)
	{
		is_piece_in_square = true;
	}
	return is_piece_in_square;
}

// Check if a given square contains a piece.
bool isPieceInSquare(Mat binary_image, int square_number)
{
	bool is_piece_in_square = false;

	int coordinates[2] = { -1, -1 };
	getSquareCoordinates(square_number, coordinates);

	int object_pixels_in_square = getNumberOfObjectPixelsInSquare(binary_image, coordinates[0], coordinates[1]);
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
	//displayImage(to_string(top_left_y), square_image);

	// Histogram hue from square.
	Mat hist = hueHistogram(square_image, 25);
	//displayHistogram(to_string(top_left_y), hist, 25);
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

// Check what colour piece is within a given square.
bool isBlackPiece(Mat rgb_image, int square_number)
{
	bool is_black_piece = false;

	int coordinates[2] = { -1, -1 };
	getSquareCoordinates(square_number, coordinates);

	// Extract square from image.
	Range rows(coordinates[0], coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS);
	Range cols(coordinates[1], coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS);
	Mat square_image = rgb_image(cols, rows);
	//displayImage(to_string(top_left_y), square_image);

	// Histogram hue from square.
	Mat hist = hueHistogram(square_image, 25);
	//displayHistogram(to_string(top_left_y), hist, 25);
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

// Check whether a given move is valid.
bool isValidMove(int previous_board[NUMBER_OF_SQUARES], int current_board[NUMBER_OF_SQUARES], int from, int to)
{
	bool isValidMove = false;

	// Check that the piece is moving to an empty space.
	/*cout << "\tCurrent Board From[" << from << "] = " << current_board[from - 1] << endl; 
	cout << "\tCurrent Board To[" << to << "] = " << current_board[to] << endl;
	cout << "\t Previous Board From[" << from << "] " << previous_board[from - 1] << endl;
	cout << "\t Previous Board To[" << to << "] " << previous_board[to - 1]  << endl;*/
	if ((previous_board[from - 1] == current_board[to - 1]) && (previous_board[to - 1] == EMPTY_SQUARE) && (current_board[from - 1] == EMPTY_SQUARE))
	{
		// Get coordinates.
		int from_coordinates[2] = { -1, -1 };
		getSquareCoordinates(from, from_coordinates);
		int to_coordinates[2] = { -1, -1 };
		getSquareCoordinates(to, to_coordinates);

		// Check moves.
		if ((from_coordinates[0] == to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) // left-up
			|| (from_coordinates[0] == to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS) // left-down
			|| (from_coordinates[0] == to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) // right-up
			|| (from_coordinates[0] == to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS)) // right-down
		{
			isValidMove = true;
		}
		else if (((from_coordinates[0] == to_coordinates[0] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) // left-up
				&& (current_board[getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
			|| ((from_coordinates[0] == to_coordinates[0] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) // left-down
				&& (current_board[getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
			|| ((from_coordinates[0] == to_coordinates[0] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) // right-up
				&& (current_board[getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
			|| ((from_coordinates[0] == to_coordinates[0] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) // right-down
				&& (current_board[getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE)))
		{
			isValidMove = true;
		}
	}
	return isValidMove;
}

// Check whether a given move is valid.
void executeMove(int previous_board[NUMBER_OF_SQUARES], int current_board[NUMBER_OF_SQUARES], int from, int to)
{
	// Make move.
	int temp = previous_board[from - 1];
	previous_board[from - 1] = previous_board[to - 1];
	previous_board[to - 1] = temp;

	// Get coordinates.
	int from_coordinates[2] = { -1, -1 };
	getSquareCoordinates(from, from_coordinates);
	int to_coordinates[2] = { -1, -1 };
	getSquareCoordinates(to, to_coordinates);

	// Delete piece if necessary.
	if ((from_coordinates[0] == to_coordinates[0] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) // left-up
		&& (current_board[getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
	{
		int square = getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS);
		previous_board[square - 1] = EMPTY_SQUARE;
		cout << "\tPiece taken at " << square << endl;
	}
	else if ((from_coordinates[0] == to_coordinates[0] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) // left-down
		&& (current_board[getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
	{
		int square = getSquare(to_coordinates[0] - SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS);
		previous_board[square - 1] = EMPTY_SQUARE;
		cout << "\tPiece taken at " << square << endl;
	}
	else if ((from_coordinates[0] == to_coordinates[0] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] - 2 * SQUARE_DIMENSIONS_IN_PIXELS) // right-up
		&& (current_board[getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
	{
		int square = getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] - SQUARE_DIMENSIONS_IN_PIXELS);
		previous_board[square - 1] = EMPTY_SQUARE;
		cout << "\tPiece taken at " << square << endl;
	}
	else if ((from_coordinates[0] == to_coordinates[0] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) && (from_coordinates[1] == to_coordinates[1] + 2 * SQUARE_DIMENSIONS_IN_PIXELS) // right-down
			&& (current_board[getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS) - 1] == EMPTY_SQUARE))
	{
		int square = getSquare(to_coordinates[0] + SQUARE_DIMENSIONS_IN_PIXELS, to_coordinates[1] + SQUARE_DIMENSIONS_IN_PIXELS);
		previous_board[square - 1] = EMPTY_SQUARE;
		cout << "\tPiece taken at " << square << endl;
	}
}

// Check whether the piece is a king or not.
bool isKing(Mat binary_image, int top_left_x, int top_left_y)
{
	bool isKing = false;
	// Extract square from image.
	Range rows(top_left_x, top_left_x + SQUARE_DIMENSIONS_IN_PIXELS);
	Range cols(top_left_y, top_left_y + SQUARE_DIMENSIONS_IN_PIXELS);
	Mat square_image = binary_image(cols, rows);
	//DisplayImage(to_string(top_left_y), square_image);

	// Find contour of piece.
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(square_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//cout << "contours length:" << contours.size() << endl;
	//drawContours(contours_image, contours, contour, colour, LINE_4, 8, hierarchy);
	/*Mat contours_image = Mat::zeros(square_image.size(), CV_8UC3);;
	drawContours(contours_image, contours, 0, (0, 255, 0), 3);
	DisplayImage(to_string(top_left_y), contours_image);*/

	// Compute circularity.
	double area = contourArea(contours[0]);
	//cout << "contour area:" << area << endl;
	double perimeter = arcLength(contours[0], true);
	//cout << "contour arc length:" << perimeter << endl;
	double circularity = (4 * (2 * acos(0.0)) * area) / (perimeter * perimeter);
	//cout << "contour circularity:" << circularity << endl;
	if (circularity < 0.80)
	{
		isKing = true;
	}
	return isKing;
}

// Check the ground truth for what the specified square should contain.
int checkBoardGroundTruth(int square_number, string white_pieces, string black_pieces)
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

// Check the ground truth for what the specified square should contain including kings.
int checkBoardGroundTruthWithKings(int square_number, string white_pieces, string black_pieces)
{
	int ground_truth = EMPTY_SQUARE;

	// Check if it should be a white man or king.
	regex rgx(",");
	sregex_token_iterator white_iterator(white_pieces.begin(), white_pieces.end(), rgx, -1);
	sregex_token_iterator white_end;
	for (; ground_truth == EMPTY_SQUARE && white_iterator != white_end; ++white_iterator)
	{
		string current_square = *white_iterator;
		if (current_square == to_string(square_number))
		{
			ground_truth = WHITE_MAN_ON_SQUARE;
		}
		else if (current_square == "K" + to_string(square_number))
		{
			ground_truth = WHITE_KING_ON_SQUARE;
		}
	}

	// Check if it should be a black man or king.
	sregex_token_iterator black_iterator(black_pieces.begin(), black_pieces.end(), rgx, -1);
	sregex_token_iterator black_end;
	for (; ground_truth == EMPTY_SQUARE && black_iterator != black_end; ++black_iterator)
	{
		string current_square = *black_iterator;
		if (current_square == to_string(square_number))
		{
			ground_truth = BLACK_MAN_ON_SQUARE;
		}
		else if (current_square == "K" + to_string(square_number))
		{
			ground_truth = BLACK_KING_ON_SQUARE;
		}
	}

	return ground_truth;
}

// Update the confusion matrix based on what was detected in the square and what was recorded in the ground truth.
void updateConfusionMatrix(int confusion_matrix[3][3], int detected_square_contents, int actual_square_contents)
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
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[0][2]++;
					//cout << "misclassification" << endl;
					break;
			}
			break;
		case(WHITE_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					confusion_matrix[1][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					confusion_matrix[1][1]++;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[1][2]++;
					//cout << "misclassification" << endl;
					break;
			}
			break;
		case(BLACK_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					confusion_matrix[2][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					confusion_matrix[2][1]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					confusion_matrix[2][2]++;
					break;
			}
			break;
	}
}

// Update the extended confusion matrix based on what was detected in the square and what was recorded in the ground truth.
void updateExtendedConfusionMatrix(int extended_confusion_matrix[5][5], int detected_square_contents, int actual_square_contents)
{
	switch (detected_square_contents)
	{
		case(EMPTY_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					extended_confusion_matrix[0][0]++;
					break;
				case(WHITE_MAN_ON_SQUARE):
					extended_confusion_matrix[0][1]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_KING_ON_SQUARE):
					extended_confusion_matrix[0][2]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					extended_confusion_matrix[0][3]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_KING_ON_SQUARE):
					extended_confusion_matrix[0][4]++;
					//cout << "misclassification" << endl;
					break;
			}
			break;
		case(WHITE_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					extended_confusion_matrix[1][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					extended_confusion_matrix[1][1]++;
					break;
				case(WHITE_KING_ON_SQUARE):
					extended_confusion_matrix[1][2]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					extended_confusion_matrix[1][3]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_KING_ON_SQUARE):
					extended_confusion_matrix[1][4]++;
					//cout << "misclassification" << endl;
					break;
			}
			break;
		case(WHITE_KING_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					extended_confusion_matrix[2][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					extended_confusion_matrix[2][1]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_KING_ON_SQUARE):
					extended_confusion_matrix[2][2]++;
					break;
				case(BLACK_MAN_ON_SQUARE):
					extended_confusion_matrix[2][3]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_KING_ON_SQUARE):
					extended_confusion_matrix[2][4]++;
					//cout << "misclassification" << endl;
					break;
			}
			break;
		case(BLACK_MAN_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					extended_confusion_matrix[3][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					extended_confusion_matrix[3][1]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_KING_ON_SQUARE):
					extended_confusion_matrix[3][2]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					extended_confusion_matrix[3][3]++;
					break;
				case(BLACK_KING_ON_SQUARE):
					extended_confusion_matrix[3][4]++;
					//cout << "misclassification" << endl;
					break;
				}
			break;
		case(BLACK_KING_ON_SQUARE):
			switch (actual_square_contents)
			{
				case(EMPTY_SQUARE):
					extended_confusion_matrix[4][0]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_MAN_ON_SQUARE):
					extended_confusion_matrix[4][1]++;
					//cout << "misclassification" << endl;
					break;
				case(WHITE_KING_ON_SQUARE):
					extended_confusion_matrix[4][2]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_MAN_ON_SQUARE):
					extended_confusion_matrix[4][3]++;
					//cout << "misclassification" << endl;
					break;
				case(BLACK_KING_ON_SQUARE):
					extended_confusion_matrix[4][4]++;
					break;
			}
			break;
	}
}

// Extract Hue channel from RGB image converted to HSV image.
Mat extractHue(Mat rgb_image)
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
Mat hueHistogram(Mat rgb_image, int bins)
{
    // Extract hue channel.
    Mat hue = extractHue(rgb_image);
    
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
void displayHistogram(string name, Mat hist, int bins)
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

// Perform Hough Transformations.
void houghTransforms(Mat board_image)
{
	// Convert to greyscale.
	Mat grey_board_image;
	cvtColor(board_image, grey_board_image, COLOR_BGR2GRAY);

	// Perform Canny edge detection.
	Mat edges_image;
	Canny(grey_board_image, edges_image, 50, 200, 3);
	edges_image = dilate(edges_image, getStructuringElement3x3());
	displayImage("Canny Edge Detection", edges_image);

	// Standard Hough Line Transform
	Mat hough_transform_image;
	cvtColor(edges_image, hough_transform_image, COLOR_GRAY2BGR);
	vector<Vec2f> lines;
	HoughLines(edges_image, lines, 1, CV_PI / 200, 255, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(hough_transform_image, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	//displayImage("Hough Line Transform", hough_transform_image);

	// Probabilistic Line Transform
	Mat hough_prob_transform_image;
	cvtColor(edges_image, hough_prob_transform_image, COLOR_GRAY2BGR);
	vector<Vec4i> linesP;
	HoughLinesP(edges_image, linesP, 1, CV_PI / 200, 20, 20, 5);
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(hough_prob_transform_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	//displayImage("Hough Probabilistic Line Transform", hough_prob_transform_image);

	Mat hough_transform_images = JoinImagesHorizontally(hough_transform_image, "Hough Line Transform", hough_prob_transform_image, "Hough Probabilistic Line Transform", 4, 255);
	displayImage("Hough Transforms", hough_transform_images);
}

// Find contours in image.
void contourFollowing(Mat board_image)
{
	// Convert to greyscale and threshold.
	Mat grey_board_image;
	cvtColor(board_image, grey_board_image, COLOR_BGR2GRAY);
	Mat thresholded_board_image;
	threshold(grey_board_image, thresholded_board_image, 127, 255, THRESH_BINARY);

	// Find contours.
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresholded_board_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// Display contours.
	Mat contours_image = Mat::zeros(thresholded_board_image.size(), CV_8UC3);;
	for (int contour = 0; (contour < contours.size()); contour++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(contours_image, contours, contour, colour, LINE_4, 8, hierarchy);
	}
	displayImage("Empty Board", contours_image);
}

// Find corners using findChessboardCorners function.
void findCorners(Mat board_image)
{
	// Convert to greyscale.
	Mat grey_board_image;
	cvtColor(board_image, grey_board_image, COLOR_BGR2GRAY);
	//displayImage("grey", grey_board_image);

	// Find chessboard corners.
	Size pattern_size(7, 7);
	vector<Point2f> chessboard_corners;
	bool patternFound = findChessboardCorners(grey_board_image, pattern_size, chessboard_corners);
	if (patternFound)
		cornerSubPix(grey_board_image, chessboard_corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
	drawChessboardCorners(board_image, pattern_size, Mat(chessboard_corners), patternFound);
	displayImage("Corners", board_image);

	// Print corners found.
	/*for (Point2f corner : chessboard_corners)
	{
		cout << corner << endl;
	}*/

	// Calculate corners.
	Point2f top_left = Point2f(chessboard_corners[48].x - (chessboard_corners[47].x - chessboard_corners[48].x),
		chessboard_corners[48].y - (chessboard_corners[41].y - chessboard_corners[48].y));
	Point2f top_right = Point2f(chessboard_corners[42].x + (chessboard_corners[42].x - chessboard_corners[43].x),
		chessboard_corners[42].y - (chessboard_corners[35].y - chessboard_corners[42].y));
	Point2f bottom_left = Point2f(chessboard_corners[6].x - (chessboard_corners[5].x - chessboard_corners[6].x),
		chessboard_corners[6].y + (chessboard_corners[6].y - chessboard_corners[13].y));
	Point2f bottom_right = Point2f(chessboard_corners[0].x + (chessboard_corners[0].x - chessboard_corners[1].x),
		chessboard_corners[0].y + (chessboard_corners[0].y - chessboard_corners[7].y));
	cout << "Calculated corners:\n" << top_left << "\t" << top_right << "\n" << bottom_left << "\t" << bottom_right << endl;

	// Calculate adjustments for corners.
	float top_left_x = (chessboard_corners[46].x - chessboard_corners[47].x) - (chessboard_corners[47].x - chessboard_corners[48].x);
	float top_left_y = (chessboard_corners[34].y - chessboard_corners[41].y) - (chessboard_corners[41].y - chessboard_corners[48].y);

	float top_right_x = (chessboard_corners[42].x - chessboard_corners[43].x) - (chessboard_corners[43].x - chessboard_corners[44].x);
	float top_right_y = (chessboard_corners[42].y - chessboard_corners[35].y) - (chessboard_corners[35].y - chessboard_corners[28].y);

	float bottom_left_x = (chessboard_corners[6].x - chessboard_corners[5].x) - (chessboard_corners[5].x - chessboard_corners[4].x);
	float bottom_left_y = (chessboard_corners[6].y - chessboard_corners[13].y) - (chessboard_corners[13].y - chessboard_corners[20].y);

	float bottom_right_x = (chessboard_corners[1].x - chessboard_corners[2].x) - (chessboard_corners[0].x - chessboard_corners[1].x);
	float bottom_right_y = (chessboard_corners[0].y - chessboard_corners[7].y) - (chessboard_corners[7].y - chessboard_corners[14].y);

	
	// Adjust corners.
	top_left = Point2f(chessboard_corners[48].x - (chessboard_corners[47].x - chessboard_corners[48].x) + top_left_x, 
								chessboard_corners[48].y - (chessboard_corners[41].y - chessboard_corners[48].y) + top_left_y);
	top_right = Point2f(chessboard_corners[42].x + (chessboard_corners[42].x - chessboard_corners[43].x) + top_right_x, 
								chessboard_corners[42].y - (chessboard_corners[35].y - chessboard_corners[42].y) + top_right_y);
	bottom_left = Point2f(chessboard_corners[6].x - (chessboard_corners[5].x - chessboard_corners[6].x) + bottom_left_x,
								chessboard_corners[6].y + (chessboard_corners[6].y - chessboard_corners[13].y) + bottom_left_y);
	bottom_right = Point2f(chessboard_corners[0].x + (chessboard_corners[0].x - chessboard_corners[1].x) + bottom_right_x,
								chessboard_corners[0].y + (chessboard_corners[0].y - chessboard_corners[7].y) + bottom_right_y);
	cout << "Calculated corners withh adjustment:\n" << top_left << "\t" << top_right << "\n" << bottom_left << "\t" << bottom_right << endl;

	// Perform perspective transform with detected corners.
	Mat result = Mat::zeros(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS, board_image.type());
	Point2f source[4] = { top_left, bottom_left, top_right, bottom_right };
	Point2f destination[4] = { Point2f(0, 0), Point2f(0, BOARD_DIMENSIONS_IN_PIXELS), Point2f(BOARD_DIMENSIONS_IN_PIXELS, 0), Point2f(BOARD_DIMENSIONS_IN_PIXELS, BOARD_DIMENSIONS_IN_PIXELS) };
	Mat perspective_matrix = getPerspectiveTransform(source, destination);
	warpPerspective(board_image, result, perspective_matrix, result.size());
	displayImage("Perspective transformation using detected corners", result);
}

// Backproject the given histogram onto the image.
Mat backproject(int, void*, Mat hist, Mat* image)
{
    // Backproject the histogram onto image.
    Mat backproj;
    float hue_range[] = { 0, 180 };
    const float* ranges[] = { hue_range };
    calcBackProject(image, 1, 0, hist, backproj, ranges, 1, true);
    return backproj;
}

// Perform backprojection.
void histogramAndBackproject(string name, Mat sample_image, Mat rgb_image, int bins) {
    // Histogram hue in sample image.
    Mat histogram = hueHistogram(sample_image, bins);

    // Print histogram.
    cout << name << " histogram:\n" << histogram << endl;

    // Show histogram.
    //displayHistogram(name, histogram, bins);

    // Extract Hue channel from RGB image.
    Mat image = extractHue(rgb_image);

    // Backproject
    Mat backprojection = backproject(0, 0, histogram, &image);

    // Print backprojection.
    //printMatrix(name + " backprojection", backprojection, 100);

    // Show backprojection.
    displayImage(name + " backprojection", backprojection);

    // Threshold.
    Mat thresholdedImage;
    threshold(backprojection, thresholdedImage, 127, 255, THRESH_BINARY);

    // Print thresholded image.
    //printMatrix(name + " thresholded", thresholdedImage, 100);

    // Show thresholded image.
    //displayImage(name + " thresholded", thresholdedImage);
}

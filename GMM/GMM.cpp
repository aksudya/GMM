#include "MOG.h"

using namespace cv;
using namespace std;


int main(int argc, const char** argv)
{
	VideoWriter aoutputVideo;
	Mat src,gray,mask;
	MOG mog;
	clock_t start, end;
	clock_t start1, end1;
	aoutputVideo.open("./demo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 4, cv::Size(320, 240), true);
	if (!aoutputVideo.isOpened())
	{
		std::cout << "Could not open the output video for write " << std::endl;
		return -1;
	}

	start = clock();
	for (int i = 0; i <= 200; ++i)
	{
		std::ostringstream os;
		os<<setw(5) << setfill('0') << i;
		string path = "./WavingTrees/b"+os.str()+".bmp";
		src = imread(path);
		
		if (i == 0)
			mog.init(src);
		else
			mog.train(src);
	}
	mog.get_B(src);
	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Total time:" << endtime * 1000 << "ms" << endl;

	for (int i = 230; i <= 260; ++i)
	{
		Mat combine, combine1, combine2;
		start1 = clock();
		std::ostringstream os;
		os << setw(5) << setfill('0') << i;
		string path = "./WavingTrees/b" + os.str() + ".bmp";
		src = imread(path);
		
		mog.test(src);
		

		mask = mog.mask.clone();
		Mat mask1= mog.mask.clone();
		Mat mask_out;
		cvtColor(mask1, mask_out, COLOR_GRAY2RGB);
		hconcat(src, mask_out, combine1);
		//morphologyEx(mask, mask, MORPH_OPEN, Mat::ones(5,5, CV_8UC1));
		erode(mask, mask, Mat::ones(5, 5, CV_8UC1), Point(-1, -1));
		dilate(mask, mask, Mat::ones(7, 7, CV_8UC1), Point(-1, -1));
		
		end1 = clock();
		double endtime1 = (double)(end1 - start1) / CLOCKS_PER_SEC;
		cout << i << " " << endtime1 * 1000 << "ms" << endl;
		
		Mat res;
		src.copyTo(res, mask);
		Mat mask_out1;
		cvtColor(mask, mask_out1, COLOR_GRAY2RGB);
		hconcat(mask_out1, res, combine2);
		vconcat(combine1, combine2, combine);

		aoutputVideo.write(combine);
	}

	



		


	//waitKey(0);
	//imshow("gray", mask);
	//imshow("gray", mask);
}
#include "MOG.h"


using namespace cv;
using namespace std;


int main(int argc, const char** argv)
{
	Mat src,gray,mask;
	MOG mog;
	for (int i = 0; i <= 100; ++i)
	{
		std::ostringstream os;
		os<<setw(5) << setfill('0') << i;
		string path = "./WavingTrees/b"+os.str()+".bmp";
		src = imread(path);
		//cvtColor(src, gray, CV_BGR2GRAY);
		if (i == 0)
			mog.init(src);
		else
			mog.train(src);
	}
	mog.get_B(src);
	for (int i = 230; i <= 260; ++i)
	{
		std::ostringstream os;
		os << setw(5) << setfill('0') << i;
		string path = "./WavingTrees/b" + os.str() + ".bmp";
		src = imread(path);
		
		//cvtColor(src, gray, CV_BGR2GRAY);
		mog.test(src);

		mask = mog.mask.clone();
		//morphologyEx(mask, mask, MORPH_OPEN, Mat());
		erode(mask, mask, Mat(5, 5, CV_8UC1), Point(-1, -1));  // You can use Mat(5, 5, CV_8UC1) here for less distortion  
		dilate(mask, mask, Mat(5, 5, CV_8UC1), Point(-1, -1));
		
	}

	imshow("gray",gray);
}
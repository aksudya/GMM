#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include <iostream>
#include <omp.h>

#define MAX_GMM 5
#define GMM_LEARN_ALPHA 0.05
#define GMM_LEARN_ROI 0.05
#define GMM_TRES 0.6
#define GMM_ACC 2.5


using namespace cv;
using namespace std;

class MOG
{
public:
	MOG();
	~MOG();

	Mat weight[MAX_GMM];
	Mat sigema[MAX_GMM];
	Mat m_mean[MAX_GMM];

	Mat mask;
	Mat B;

	void init(Mat first_img);
	void train(Mat img);
	void sort_M(int i,int j);
	void get_B(Mat img);
	void test(Mat img);

};

class gauss
{
public:
	float weight;
	float sigema;
	int mean;
	float rank;

	gauss()
	{
		weight = 0;
		sigema = 15;
		mean = 0;
		rank = 0;
	}

	~gauss()
	{
		
	}

	gauss(float weight,float sigema,int mean)
	{
		this->weight = weight;
		this->sigema = sigema;
		this->mean = mean;
		this->rank = weight / sigema;
	}

	bool operator < (const gauss& a)
	{
		if (this->rank > a.rank)
			return true;
		else
			return false;
	}
};
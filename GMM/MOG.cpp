#include "MOG.h"

MOG::MOG()
{
}

MOG::~MOG()
{
}


void MOG::init(Mat first_img)
{
	for (int i = 0; i < MAX_GMM; i++)
	{
		weight[i] = Mat::zeros(first_img.size(), CV_32FC1);
		m_mean[i] = Mat::zeros(first_img.size(), CV_8UC1);
		sigema[i] = Mat::zeros(first_img.size(), CV_32FC1);
		sigema[i].setTo(15.0);
	}
	weight[0].setTo(1);
	m_mean[0] = first_img.clone();
	mask = Mat::zeros(first_img.size(), CV_8UC1);
	mask.setTo(255);
	B= Mat::zeros(first_img.size(), CV_8UC1);
}

void MOG::train(Mat img)
{

#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			float sum_weigt = 0;
			int match = 0;
			for (int k = 0; k < MAX_GMM; ++k)
			{
				int dis = abs(img.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j));
				dis = dis * dis;
				if(dis <= GMM_ACC * GMM_ACC *sigema[k].at<float>(i,j))		//match
				{
					weight[k].at<float>(i, j) = (1 - GMM_LEARN_ALPHA) * weight[k].at<float>(i, j) + GMM_LEARN_ALPHA;
					m_mean[k].at<uchar>(i, j) = (1 - GMM_LEARN_ROI) * (float)m_mean[k].at<uchar>(i, j) + GMM_LEARN_ROI * (float)img.at<uchar>(i, j);
					sigema[k].at<float>(i, j) = (1 - GMM_LEARN_ROI) * sigema[k].at<float>(i, j) + GMM_LEARN_ROI * dis;
					match++;
				}
				else										//not match		
				{
					weight[k].at<float>(i, j) = (1 - GMM_LEARN_ALPHA) * weight[k].at<float>(i, j);
				}
				sum_weigt += weight[k].at<float>(i, j);
			}
			for (int k = 0; k < MAX_GMM; ++k)//uniform
			{
				weight[k].at<float>(i, j) *= 1 / sum_weigt;
			}
			sort_M(i, j);
			if(match==0)			//no match
			{
				weight[MAX_GMM - 1].at<float>(i, j) = GMM_LEARN_ALPHA;
				m_mean[MAX_GMM - 1].at<uchar>(i, j) = img.at<uchar>(i, j);
				sigema[MAX_GMM - 1].at<float>(i, j) = 15;
				float sum_weight_no_match = 0;
				for (int kk = 0; kk < MAX_GMM; ++kk)
				{
					sum_weight_no_match += weight[kk].at<float>(i, j);
				}
				for (int kk = 0; kk < MAX_GMM; ++kk)
				{
					weight[kk].at<float>(i, j) *= 1 / sum_weight_no_match;
				}
				sort_M(i, j);
			}

		}
	}
}

void MOG::get_B(Mat img)
{
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			float sum_w = 0.0;   
			for (uchar k = 0; k < MAX_GMM; k++)
			{
				sum_w += weight[k].at<float>(i, j);
				if (sum_w >= GMM_TRES)  
				{
					B.at<uchar>(i, j) = k + 1;
					break;
				}
			}
		}
	}
}

void MOG::test(Mat img)
{
	mask.setTo(255);
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{		
			for (int k = 0; k < B.at<uchar>(i, j); k++)
			{
				int dis = abs(img.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j));
				dis = dis * dis;
				if (dis < GMM_ACC * GMM_ACC * sigema[k].at<float>(i, j))
				{
					mask.at<uchar>(i, j) = 0;
				}
			}
		}
	}
}

void MOG::sort_M(int i,int j)
{
	gauss gauss_arr[MAX_GMM];

	for (int k = 0; k < MAX_GMM; ++k)
	{
		gauss temp(weight[k].at<float>(i, j), sigema[k].at<float>(i, j), m_mean[k].at<uchar>(i, j));
		gauss_arr[k]=temp;
	}
	sort(gauss_arr,gauss_arr+ MAX_GMM);

	for (int k = 0; k < MAX_GMM; ++k)
	{
		weight[k].at<float>(i, j) = gauss_arr[k].weight;
		sigema[k].at<float>(i, j) = gauss_arr[k].sigema;
		m_mean[k].at<uchar>(i, j) = gauss_arr[k].mean;
	}

	
}





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
		weight[i] = Mat::zeros(first_img.size(), CV_32FC3);
		m_mean[i] = Mat::zeros(first_img.size(), CV_8UC3);
		sigema[i] = Mat::zeros(first_img.size(), CV_32FC3);
		sigema[i].setTo(15);
	}
	weight[0].setTo(1);
	m_mean[0] = first_img.clone();
	mask = Mat::zeros(first_img.size(), CV_8UC1);
	mask.setTo(255);
	B = Mat::zeros(first_img.size(), CV_8UC3);
}

void MOG::train(Mat img)
{

#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			for (int channel = 0; channel < 3; ++channel)
			{


				float sum_weigt = 0;
				int match = 0;
				for (int k = 0; k < MAX_GMM; ++k)
				{
					int dis = abs(img.at<Vec3b>(i, j)[channel] - m_mean[k].at<Vec3b>(i, j)[channel]);
					dis = dis * dis;
					if (dis <= GMM_ACC * GMM_ACC * sigema[k].at<float>(i, j))		//match
					{
						weight[k].at<Vec3f>(i, j)[channel] = (1 - GMM_LEARN_ALPHA) * weight[k].at<Vec3f>(i, j)[channel] + GMM_LEARN_ALPHA;
						m_mean[k].at<Vec3b>(i, j)[channel] = (1 - GMM_LEARN_ROI) * (float)m_mean[k].at<Vec3b>(i, j)[channel] + GMM_LEARN_ROI * (float)img.at<Vec3b>(i, j)[channel];
						sigema[k].at<Vec3f>(i, j)[channel] = (1 - GMM_LEARN_ROI) * sigema[k].at<Vec3f>(i, j)[channel] + GMM_LEARN_ROI * dis;
						match++;
					}
					else										//not match		
					{
						weight[k].at<Vec3f>(i, j)[channel] = (1 - GMM_LEARN_ALPHA) * weight[k].at<Vec3f>(i, j)[channel];
					}
					sum_weigt += weight[k].at<Vec3f>(i, j)[channel];
				}
				for (int k = 0; k < MAX_GMM; ++k)//uniform
				{
					weight[k].at<Vec3f>(i, j)[channel] *= 1 / sum_weigt;
				}
				sort_M(i, j, channel);
				if (match == 0)			//no match
				{
					weight[MAX_GMM - 1].at<Vec3f>(i, j)[channel] = GMM_LEARN_ALPHA;
					m_mean[MAX_GMM - 1].at<Vec3b>(i, j)[channel] = img.at<Vec3b>(i, j)[channel];
					sigema[MAX_GMM - 1].at<Vec3f>(i, j)[channel] = 15;
					float sum_weight_no_match = 0;
					for (int kk = 0; kk < MAX_GMM; ++kk)
					{
						sum_weight_no_match += weight[kk].at<Vec3f>(i, j)[channel];
					}
					for (int kk = 0; kk < MAX_GMM; ++kk)
					{
						weight[kk].at<Vec3f>(i, j)[channel] *= 1 / sum_weight_no_match;
					}
					sort_M(i, j, channel);
				}
			}

		}
	}
}

void MOG::get_B(Mat img)
{
	B.setTo(0);
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			for (int channel = 0; channel < 3; ++channel)
			{
				float sum_w = 0.0;
				for (uchar k = 0; k < MAX_GMM; k++)
				{
					sum_w += weight[k].at<Vec3f>(i, j)[channel];
					if (sum_w >= GMM_TRES)
					{
						B.at<Vec3b>(i, j)[channel] = k + 1;
						break;
					}
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
			int count = 0;
			for (int channel = 0; channel < 3; ++channel)
			{
				for (int k = 0; k < B.at<Vec3b>(i, j)[channel]; k++)
				{
					int dis = abs(img.at<Vec3b>(i, j)[channel] - m_mean[k].at<Vec3b>(i, j)[channel]);
					dis = dis * dis;
					if (dis < GMM_ACC * GMM_ACC * sigema[k].at<Vec3f>(i, j)[channel])
					{
						count++;
						break;
					}
				}
			}
			if(count==3)
			{
				mask.at<uchar>(i, j) = 0;
			}
		}
	}
}

void MOG::sort_M(int i, int j,int channel)
{
	gauss gauss_arr[MAX_GMM];

	for (int k = 0; k < MAX_GMM; ++k)
	{
		gauss temp(weight[k].at<Vec3f>(i, j)[channel], sigema[k].at<Vec3f>(i, j)[channel], m_mean[k].at<Vec3b>(i, j)[channel]);
		gauss_arr[k] = temp;
	}
	sort(gauss_arr, gauss_arr + MAX_GMM);

	for (int k = 0; k < MAX_GMM; ++k)
	{
		weight[k].at<Vec3f>(i, j)[channel] = gauss_arr[k].weight;
		sigema[k].at<Vec3f>(i, j)[channel] = gauss_arr[k].sigema;
		m_mean[k].at<Vec3b>(i, j)[channel] = gauss_arr[k].mean;
	}


}





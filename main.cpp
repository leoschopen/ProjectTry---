#include "morphology.h"

int erosionElem = MORPH_RECT;
int erosionSize = 1;
int dilationElem = MORPH_RECT;
int dilationSize = 1;
int const maxElem = 2;
int const maxKernelSize = 21;

Mat srcImg, dstImg, binaryImg, erosionDst, dilationDst;

// 腐蚀操作
void Erosion()
{
	int elemType = MORPH_RECT;
	Mat element = getStructuringElement(elemType,
		Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		Point(erosionSize, erosionSize));


	erode(binaryImg, erosionDst, element);

	namedWindow("Original Image", WINDOW_NORMAL);
	imshow("Original Image", srcImg);
	namedWindow("Erosion", WINDOW_NORMAL);
	imshow("Erosion", erosionDst);
}

void binaryDilate() {
	int elemType = MORPH_RECT;

	Mat element = getStructuringElement(elemType,
		Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		Point(dilationSize, dilationSize));
	//膨胀操作
	dilate(binaryImg, dilationDst, element);
	namedWindow("Original Image", WINDOW_NORMAL);
	imshow("Original Image", srcImg);
	namedWindow("Dilated", WINDOW_NORMAL);
	imshow("Dilated", dilationDst);
}

void binaryEx_open() {
	Mat element5(5, 5, CV_8U, Scalar(1));
	morphologyEx(binaryImg, dstImg, MORPH_OPEN, element5);
	namedWindow("Original Image");
	imshow("Original Image", srcImg);

	namedWindow("Result Image");
	imshow("Result Image", dstImg);
}

void binaryEx_close() {
	Mat element5(5, 5, CV_8U, Scalar(1));
	morphologyEx(binaryImg, dstImg, MORPH_CLOSE, element5);
	namedWindow("Original Image");
	imshow("Original Image", srcImg);
	namedWindow("Result Image");
	imshow("Result Image", dstImg);
}

Mat binarySkeletonization() {
	int i, j, k;
	uchar p[11];
	int pos[9][2] = { {0,0}, {-1,0} , {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1} };
	int cond1, cond2, cond3, cond4, counter = 0;
	bool pointsDeleted = true;
	Mat mask, dstImg;

	//转化为0,1二值图像
	dstImg = binaryImg / 255;
	//若没有可删除像素点
	while (pointsDeleted)
	{
		mask = Mat::zeros(dstImg.size(), CV_8UC1);//初始化模板为全0
		pointsDeleted = false;
		for (i = 1; i < dstImg.rows - 1; i++)
		{
			for (j = 1; j < dstImg.cols - 1; j++)
			{
				//获取3*3结构元素p1~p9对应像素值, 其中p1为中心点
				for (k = 1; k < 10; k++)
					p[k] = dstImg.at<uchar>(i + pos[k - 1][0], j + pos[k - 1][1]);

				//若中心点为背景色，跳过
				if (p[1] == 0) continue;

				//计算中心点周围所有像素值之和
				cond1 = 0;
				for (k = 2; k < 10; k++) cond1 += p[k];

				//计算p2~p9从0到1变化的次数
				cond2 = 0;
				p[10] = p[2]; //用于处理k=8时, p[k+2]越界情况
				for (k = 2; k < 10; k += 2)
					cond2 += ((p[k] == 0 && p[k + 1] == 1) + (p[k + 1] == 0 && p[k + 2] == 1));

				if (counter % 2 == 0)//偶数次迭代判断条件
				{
					cond3 = p[2] * p[4] * p[6];
					cond4 = p[4] * p[6] * p[8];
				}
				else//奇数次迭代判断条件
				{
					cond3 = p[2] * p[4] * p[8];
					cond4 = p[2] * p[6] * p[8];
				}
				//若同时满足条件1~条件4
				if ((2 <= cond1 && cond1 <= 6) && (cond2 == 1) && (cond3 == 0) && (cond4 == 0))
				{
					pointsDeleted = true;
					mask.at<uchar>(i, j) = 1; //写入待删除的像素点至模板
				}
			}
		}
		dstImg &= ~mask; //通过逻辑与操作删除目标像素点
		counter++;
	}
	//恢复为0, 255二值图像
	dstImg *= 255;

	return dstImg;
}

void drawCircles(const Mat& maskImg, Mat& dstImg)
{
	uchar pixel;
	for (int i = 0; i < maskImg.rows; i++)
		for (int j = 0; j < maskImg.cols; j++)
		{
			pixel = maskImg.at<uchar>(i, j);
			if (pixel == 255)
				circle(dstImg, Point(j, i), 5, Scalar(255, 255, 255));
		}
}

Mat findCorners(const Mat& inImg) {
	int i, elemSize = 5;
	Mat dstImg1, dstImg2, diffImg;

	//设置十字形结构元素
	Mat crossStruct = getStructuringElement(MORPH_CROSS,
		Size(elemSize, elemSize),
		Point(elemSize / 2, elemSize / 2));
	//设置矩形结构元素
	Mat rectStruct = getStructuringElement(MORPH_RECT,
		Size(elemSize, elemSize),
		Point(elemSize / 2, elemSize / 2));
	//自定义X形结构元素
	Mat xStruct(elemSize, elemSize, CV_8U, Scalar(0));
	for (i = 0; i < elemSize; i++)
	{
		xStruct.at<uchar>(i, i) = 1;
		xStruct.at<uchar>(4 - i, i) = 1;
	}
	//自定义菱形结构元素
	Mat diamondStruct(elemSize, elemSize, CV_8U, Scalar(1));
	diamondStruct.at<uchar>(0, 0) = 0;
	diamondStruct.at<uchar>(0, 1) = 0;
	diamondStruct.at<uchar>(1, 0) = 0;
	diamondStruct.at<uchar>(4, 4) = 0;
	diamondStruct.at<uchar>(3, 4) = 0;
	diamondStruct.at<uchar>(4, 3) = 0;
	diamondStruct.at<uchar>(4, 0) = 0;
	diamondStruct.at<uchar>(4, 1) = 0;
	diamondStruct.at<uchar>(3, 0) = 0;
	diamondStruct.at<uchar>(0, 4) = 0;
	diamondStruct.at<uchar>(0, 3) = 0;
	diamondStruct.at<uchar>(1, 4) = 0;

	//用十字形结构元素膨胀图像
	dilate(inImg, dstImg1, crossStruct);
	//用菱形结构元素腐蚀图像
	erode(dstImg1, dstImg1, diamondStruct);

	//用X形结构元素膨胀图像
	dilate(inImg, dstImg2, xStruct);
	//用矩形结构元素腐蚀图像
	erode(dstImg2, dstImg2, rectStruct);

	absdiff(dstImg2, dstImg1, diffImg);

	threshold(diffImg, dstImg1, 60, 255, THRESH_BINARY);

	dstImg2 = inImg;

	drawCircles(dstImg1, dstImg2);

	return dstImg2;
}

int main(int argc, char** argv)

{
	srcImg = imread("D:\\STUDY\\DIP\\实验\\lab5\\1.jpg");

	Mat grayImg(srcImg.size(), CV_8U);
	//将源图像转化为灰度图像
	cvtColor(srcImg, grayImg, CV_BGR2GRAY);

	binaryImg = grayImg;

	//二值化处理
	threshold(grayImg, binaryImg, 100, 255, THRESH_BINARY);

	while (1) {
		system("cls");
		cout << "\t\t\t==========================================" << endl;
		cout << "\t\t\t***  1 ----    二值形态学腐蚀     ----     ***" << endl;
		cout << "\t\t\t***  2 ----    二值形态学膨胀     ----     ***" << endl;
		cout << "\t\t\t***  3 ----    二值形态学开运算   ----     ***" << endl;
		cout << "\t\t\t***  4 ----    二值形态学闭运算   ----     ***" << endl;
		cout << "\t\t\t***  5 ----    二值形态学骨架提取 ----     ***" << endl;
		cout << "\t\t\t***  6 ----    形态学检测角点     ----     ***" << endl;
		cout << "\t\t\t==========================================" << endl;
		cout << "\n\t\t\t   请输入您的选择：";
		int choice;
		while (1)
		{
			cin >> choice;
			cin.clear();
			cin.sync();//清空cin缓冲区里面未读取的信息
			if (choice < 1 || choice>10)
				cout << "您的输入有误，请重新输入：";
			else
				break;
		}
		switch (choice)
		{
		case 1: {
			// 创建显示窗口

			Erosion();
			waitKey(0);
			break;
		}
			  
		case 2: {

			binaryDilate();
			waitKey(0);
			break;
		}

		case 3: {
			binaryEx_open();
			waitKey(0);
			break;
		}

		case 4: {
			binaryEx_close();
			waitKey(0);
			break;
		}

		case 5: {
			//如果细化目标为黑色区域，则反转图像
			grayImg = 255 - grayImg;

			//二值化处理
			threshold(grayImg, binaryImg, 100, 255, THRESH_BINARY);

			//细化处理
			dstImg = binarySkeletonization();

			dstImg = 255 - dstImg;

			namedWindow("Original Image");
			imshow("Original Image", srcImg);

			namedWindow("Result Image");
			imshow("Result Image", dstImg);
			waitKey(0);
			break;
		}
		case 6: {
			binaryImg = grayImg;

			dstImg = findCorners(binaryImg);

			namedWindow("Original Image");
			imshow("Original Image", srcImg);

			namedWindow("Result Image");
			imshow("Result Image", dstImg);
			waitKey(0);
			break;
		}
		}
	}
	return 0;
}
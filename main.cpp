#include "morphology.h"

int erosionElem = MORPH_RECT;
int erosionSize = 1;
int dilationElem = MORPH_RECT;
int dilationSize = 1;
int const maxElem = 2;
int const maxKernelSize = 21;

Mat srcImg, dstImg, binaryImg, erosionDst, dilationDst;

// ��ʴ����
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
	//���Ͳ���
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

	//ת��Ϊ0,1��ֵͼ��
	dstImg = binaryImg / 255;
	//��û�п�ɾ�����ص�
	while (pointsDeleted)
	{
		mask = Mat::zeros(dstImg.size(), CV_8UC1);//��ʼ��ģ��Ϊȫ0
		pointsDeleted = false;
		for (i = 1; i < dstImg.rows - 1; i++)
		{
			for (j = 1; j < dstImg.cols - 1; j++)
			{
				//��ȡ3*3�ṹԪ��p1~p9��Ӧ����ֵ, ����p1Ϊ���ĵ�
				for (k = 1; k < 10; k++)
					p[k] = dstImg.at<uchar>(i + pos[k - 1][0], j + pos[k - 1][1]);

				//�����ĵ�Ϊ����ɫ������
				if (p[1] == 0) continue;

				//�������ĵ���Χ��������ֵ֮��
				cond1 = 0;
				for (k = 2; k < 10; k++) cond1 += p[k];

				//����p2~p9��0��1�仯�Ĵ���
				cond2 = 0;
				p[10] = p[2]; //���ڴ���k=8ʱ, p[k+2]Խ�����
				for (k = 2; k < 10; k += 2)
					cond2 += ((p[k] == 0 && p[k + 1] == 1) + (p[k + 1] == 0 && p[k + 2] == 1));

				if (counter % 2 == 0)//ż���ε����ж�����
				{
					cond3 = p[2] * p[4] * p[6];
					cond4 = p[4] * p[6] * p[8];
				}
				else//�����ε����ж�����
				{
					cond3 = p[2] * p[4] * p[8];
					cond4 = p[2] * p[6] * p[8];
				}
				//��ͬʱ��������1~����4
				if ((2 <= cond1 && cond1 <= 6) && (cond2 == 1) && (cond3 == 0) && (cond4 == 0))
				{
					pointsDeleted = true;
					mask.at<uchar>(i, j) = 1; //д���ɾ�������ص���ģ��
				}
			}
		}
		dstImg &= ~mask; //ͨ���߼������ɾ��Ŀ�����ص�
		counter++;
	}
	//�ָ�Ϊ0, 255��ֵͼ��
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

	//����ʮ���νṹԪ��
	Mat crossStruct = getStructuringElement(MORPH_CROSS,
		Size(elemSize, elemSize),
		Point(elemSize / 2, elemSize / 2));
	//���þ��νṹԪ��
	Mat rectStruct = getStructuringElement(MORPH_RECT,
		Size(elemSize, elemSize),
		Point(elemSize / 2, elemSize / 2));
	//�Զ���X�νṹԪ��
	Mat xStruct(elemSize, elemSize, CV_8U, Scalar(0));
	for (i = 0; i < elemSize; i++)
	{
		xStruct.at<uchar>(i, i) = 1;
		xStruct.at<uchar>(4 - i, i) = 1;
	}
	//�Զ������νṹԪ��
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

	//��ʮ���νṹԪ������ͼ��
	dilate(inImg, dstImg1, crossStruct);
	//�����νṹԪ�ظ�ʴͼ��
	erode(dstImg1, dstImg1, diamondStruct);

	//��X�νṹԪ������ͼ��
	dilate(inImg, dstImg2, xStruct);
	//�þ��νṹԪ�ظ�ʴͼ��
	erode(dstImg2, dstImg2, rectStruct);

	absdiff(dstImg2, dstImg1, diffImg);

	threshold(diffImg, dstImg1, 60, 255, THRESH_BINARY);

	dstImg2 = inImg;

	drawCircles(dstImg1, dstImg2);

	return dstImg2;
}

int main(int argc, char** argv)

{
	srcImg = imread("D:\\STUDY\\DIP\\ʵ��\\lab5\\1.jpg");

	Mat grayImg(srcImg.size(), CV_8U);
	//��Դͼ��ת��Ϊ�Ҷ�ͼ��
	cvtColor(srcImg, grayImg, CV_BGR2GRAY);

	binaryImg = grayImg;

	//��ֵ������
	threshold(grayImg, binaryImg, 100, 255, THRESH_BINARY);

	while (1) {
		system("cls");
		cout << "\t\t\t==========================================" << endl;
		cout << "\t\t\t***  1 ----    ��ֵ��̬ѧ��ʴ     ----     ***" << endl;
		cout << "\t\t\t***  2 ----    ��ֵ��̬ѧ����     ----     ***" << endl;
		cout << "\t\t\t***  3 ----    ��ֵ��̬ѧ������   ----     ***" << endl;
		cout << "\t\t\t***  4 ----    ��ֵ��̬ѧ������   ----     ***" << endl;
		cout << "\t\t\t***  5 ----    ��ֵ��̬ѧ�Ǽ���ȡ ----     ***" << endl;
		cout << "\t\t\t***  6 ----    ��̬ѧ���ǵ�     ----     ***" << endl;
		cout << "\t\t\t==========================================" << endl;
		cout << "\n\t\t\t   ����������ѡ��";
		int choice;
		while (1)
		{
			cin >> choice;
			cin.clear();
			cin.sync();//���cin����������δ��ȡ����Ϣ
			if (choice < 1 || choice>10)
				cout << "���������������������룺";
			else
				break;
		}
		switch (choice)
		{
		case 1: {
			// ������ʾ����

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
			//���ϸ��Ŀ��Ϊ��ɫ������תͼ��
			grayImg = 255 - grayImg;

			//��ֵ������
			threshold(grayImg, binaryImg, 100, 255, THRESH_BINARY);

			//ϸ������
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
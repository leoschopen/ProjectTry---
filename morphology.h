#pragma once
#include "opencv2/core/core.hpp"
#include<opencv2/imgproc/types_c.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <iostream>


using namespace cv;
using namespace std;

void Erosion(Mat binaryImg, Mat erosionDst);
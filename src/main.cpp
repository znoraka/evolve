#include "opencv2/opencv.hpp"
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <string>
#include <ctime>
#include <vector>

using namespace cv;

struct Image
{
  Mat mat;
  float psnr;
};

const int POPULATION = 100;
const int BLOCK_SIZE = 50;
Mat image;
std::vector<Image> images;
std::vector<Mat> blocks;
std::vector<std::vector<Image> > image_blocks;
int n = 0;

bool tiles = false;


RNG rng(12345);

double computePsnr(Mat im1, Mat im2) {
  double mse = 0;

  for (int i = 0; i < im1.rows; i++) {
    for (int j = 0; j < im1.cols; j++) {
      auto p1 = im1.at<Vec<uchar, 3> >(i, j);
      auto p2 = im2.at<Vec<uchar, 3> >(i, j);
      mse += pow(p1[0] - p2[0], 2);
      mse += pow(p1[1] - p2[1], 2);
      mse += pow(p1[2] - p2[2], 2);
    }
  }
  mse /= im1.rows * im1.cols * 3;

  mse = 10 * log10(pow(255, 2) /  mse);

  return mse;
}


void mutate(Image& img, Mat source) {
  Mat tmp = img.mat.clone();
  // auto a = rng.uniform(0,255);
  // auto c = Scalar(a, a, a);

  auto c = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

  if(rand() % 2 == 0) {
    Point p = Point(rand() % img.mat.rows,
		    rand() % img.mat.cols);
    int radius = rand() % ((int) ((img.mat.rows + img.mat.cols) * 0.5));
    circle(tmp, p, radius * 0.5 + ((img.mat.rows + img.mat.cols) * 0.05), c, CV_FILLED);
  } else {
    Point p1 = Point(rand() % img.mat.rows,
		     rand() % img.mat.cols);
    Point p2 = Point(rand() % img.mat.rows,
		     rand() % img.mat.cols);
    line(tmp, p1, p2, c, rand() % 50 + ((img.mat.rows + img.mat.cols) * 0.05));
  }
  
  float alpha = 0.2;
  cv::addWeighted(tmp, alpha, img.mat, 1 - alpha, 0, img.mat);
  img.psnr = computePsnr(img.mat, source);
}

void nextGen() {
  float f = 0.1f;
  float alpha = 0.9f;

  if(tiles) {
    for (int i = 0; i < image_blocks.size(); i++) {
      for (int j = 1; j < image_blocks[i].size(); j++) {
	mutate(image_blocks[i][j], blocks[i]);
      }
    }
  

    for(auto& i : image_blocks) {
      std::sort(i.begin(), i.end(), [](Image img1, Image img2) {
	  return img1.psnr > img2.psnr;
	});
    }

    for(auto& i : image_blocks) {
      for (int j = i.size() * f; j < i.size(); j++) {
	cv::addWeighted(i[rand() % i.size() * f].mat, alpha, i[j].mat, 1 - alpha, 0, i[j].mat);
      }
    }

  } else {
    for (int i = 1; i < images.size(); i++) {
      mutate(images[i], image);
    }

    std::sort(images.begin(), images.end(), [](Image img1, Image img2) {
	return img1.psnr > img2.psnr;
      });

    for (int i = images.size() * f; i < images.size(); i++) {
      cv::addWeighted(images[rand() % images.size() * f].mat, alpha, images[i].mat, 1 - alpha, 0, images[i].mat);
    }
  }
}

Mat recreateImage() {
  Mat m = image.clone();
  int n = 0;
  for (int i = 0; i < image.rows / BLOCK_SIZE; i++) {
    for (int j = 0; j < image.cols / BLOCK_SIZE; j++) {
      Mat small = image_blocks[n++][0].mat;
      for (int x = 0; x < small.rows; x++) {
      	for (int y = 0; y < small.cols; y++) {
      	  m.at<Vec<uchar, 3> >(j * BLOCK_SIZE + x, i * BLOCK_SIZE + y) = small.at<Vec<uchar, 3> >(x, y);
      	}
      }
    }
  }
  return m;
}

int main(int argc, char** argv) {
  srand (time(NULL));

  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_8UC3);

  tiles = (atoi(argv[2]) == 1);
  std::cout << tiles << std::endl;

  if(! image.data ) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  if(tiles) {
    for (int i = 0; i < image.rows / BLOCK_SIZE; i++) {
      for (int j = 0; j < image.cols / BLOCK_SIZE; j++) {
        blocks.push_back(image(Rect(i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)));
      }
    }

    for(auto j : blocks) {
      std::vector<Image> v;
      for (int i = 0; i < POPULATION; i++) {
        Image img;
        img.mat = Mat::zeros(j.size(), CV_8UC3);
        v.push_back(img);
      }
      image_blocks.push_back(v);
    }
  } else {
    for (int i = 0; i < POPULATION; i++) {
      Image img;
      img.mat = Mat::zeros(image.size(), CV_8UC3);
      images.push_back(img);
    }

  }
  // namedWindow( "Display window", WINDOW_AUTOSIZE );

  int a = 100;

  for(;;) {
    n++;
    nextGen();
    if(n % a == 0) {
      if(tiles) {
	std::string s = "images9/img" + std::to_string(n / a) + ".jpg";
	imwrite(s.c_str(), recreateImage());
      } else {
	std::string s = "images9/img" + std::to_string(n / a) + "_" + std::to_string(images[0].psnr) + ".jpg";
	imwrite(s.c_str(), images[0].mat);
      }
    }
      
    // }
    // imshow( "Display window", recreateImage());
    // std::cout << images[0].psnr << "\n";
    // if( waitKey(1) == 27 ) break;
  }

  // circle(image, Point(0,0), 50, Scalar(255,255,255), CV_FILLED, 8,0);

  // waitKey(0);

  return 0;
}

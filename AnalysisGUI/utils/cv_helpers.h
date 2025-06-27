#ifndef CV_HELPERS_H
#define CV_HELPERS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

// Function to find interior contours in an image
void find_interior(
    const double* img_data,
    const unsigned char* mask_data,
    int rows, int cols,
    double min_area_ratio,
    double max_area_ratio,
    std::vector<std::vector<cv::Point>>& contours_out,
    std::vector<uint8_t>& interior_mask_out
);

#endif // CV_HELPERS_H


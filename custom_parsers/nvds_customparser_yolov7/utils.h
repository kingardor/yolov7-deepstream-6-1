#ifndef __UTILS_H__
#define __UTILS_H__

#include <map>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

#include "NvInfer.h"

std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName, bool verbose = true);
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
std::string dimsToString(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
void printLayerInfo(
    std::string layerIndex, std::string layerName, std::string layerInput,  std::string layerOutput, std::string weightPtr);

#endif

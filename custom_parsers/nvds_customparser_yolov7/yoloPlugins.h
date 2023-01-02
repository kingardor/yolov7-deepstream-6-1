#ifndef __YOLO_PLUGINS__
#define __YOLO_PLUGINS__

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

#include <vector>

#include "NvInferPlugin.h"

#include "yolo.h"

#define CUDA_CHECK(status)                                                                                         \
    {                                                                                                              \
        if (status != 0)                                                                                           \
        {                                                                                                          \
            std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__  << " at line " \
                      << __LINE__ << std::endl;                                                                    \
            abort();                                                                                               \
        }                                                                                                          \
    }

namespace
{
const char* YOLOLAYER_PLUGIN_VERSION {"1"};
const char* YOLOLAYER_PLUGIN_NAME {"YoloLayer_TRT"};
} // namespace

class YoloLayer : public nvinfer1::IPluginV2
{
public:
    YoloLayer (const void* data, size_t length);

    YoloLayer (
        const uint& netWidth, const uint& netHeight, const uint& numClasses, const uint& newCoords,
        const std::vector<TensorInfo>& yoloTensors, const uint64_t& outputSize, const uint& modelType,
        const float& scoreThreshold);

    const char* getPluginType () const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

    const char* getPluginVersion () const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

    int getNbOutputs () const noexcept override { return 4; }

    nvinfer1::Dims getOutputDimensions (
        int index, const nvinfer1::Dims* inputs,
        int nbInputDims) noexcept override;

    bool supportsFormat (
        nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override;

    void configureWithFormat (
        const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) noexcept override;

    int initialize () noexcept override { return 0; }

    void terminate () noexcept override {}

    size_t getWorkspaceSize (int maxBatchSize) const noexcept override { return 0; }

    int32_t enqueue (
        int batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
        noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize (void* buffer) const noexcept override;

    void destroy () noexcept override { delete this; }

    nvinfer1::IPluginV2* clone() const noexcept override;

    void setPluginNamespace (const char* pluginNamespace) noexcept override {
        m_Namespace = pluginNamespace;
    }

    virtual const char* getPluginNamespace () const noexcept override {
        return m_Namespace.c_str();
    }

private:
    std::string m_Namespace {""};
    uint m_NetWidth {0};
    uint m_NetHeight {0};
    uint m_NumClasses {0};
    uint m_NewCoords {0};
    std::vector<TensorInfo> m_YoloTensors;
    uint64_t m_OutputSize {0};
    uint m_Type {0};
    float m_ScoreThreshold {0};
};

class YoloLayerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    YoloLayerPluginCreator () {}

    ~YoloLayerPluginCreator () {}

    const char* getPluginName () const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

    const char* getPluginVersion () const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        std::cerr<< "YoloLayerPluginCreator::getFieldNames is not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2* createPlugin (
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        std::cerr<< "YoloLayerPluginCreator::getFieldNames is not implemented";
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin (
        const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        std::cout << "Deserialize yoloLayer plugin: " << name << std::endl;
        return new YoloLayer(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override {
        m_Namespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override {
        return m_Namespace.c_str();
    }

private:
    std::string m_Namespace {""};
};

extern uint kNUM_CLASSES;

#endif // __YOLO_PLUGINS__

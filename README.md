# Analysis-of-workflow-of-NVDLA
Source code reading
===================
## NVDLA Compiler
### Default parameters setting 

  ```
  Usage: ./nvdla_compiler [options] --prototxt <prototxt_file> --caffemodel <caffemodel_file>
  where options include:
  -h                                              print this help message
  -P                                              project
  -i <inputpath>                                  inputPath
  -o <outputpath>                                 outputs wisdom files in 'outputpath' directory
  -t                                              testname
  --prototxt                                      prototxt file
  --caffemodel
  --cachemodel
  --profileName
  --profilecfg                                    profile from file
  --profile                                       computation profile Name (default: fast-math)
  --cprecision <fp16|int8>                        compute precision (default: int8)
  --configtarget <nv_full|nv_large|nv_small>      target platform (default: nv_full)
  --calibtable <int8 calibration table>           calibration table for INT8 networks (default: 0.00787)
  --quantizationMode <per-kernel|per-filter>      quantization mode for INT8 (default: NONE)
  --batch                                         batch size (default: 0)
  --informat <ncxhwx|nchw|nhwc>                   input data format (default: nhwc)
  ```

### Workflow of Compiler

1. important structures 

  ```c++
  struct TestInfo{
    // common
    nvdla::IWisdom* wisdom;
    std::string wisdomPath;
    // parse
    std::string modelsPath;
    std::string profilesPath;
    std::string calibTablesPath;
    // runtime
    nvdla::IRuntime* runtime;
    nvdla::ILoadable* compiledLoadable;
    NvU8 *pData;
    std::string inputImagesPath;
    std::string inputLoadablePath;
    std::map<std::string, NvDlaImage*> inputImages;
    std::map<std::string, void *> inputBuffers;
    std::map<std::string, NvDlaImage*> outputImages;
    std::map<std::string, void *> outputBuffers;
    std::vector<SubmitContext*> submits;
    NvU32 timeout;
    NvU16 numBatches; // runtime's point-of-view
    NvU32 numSubmits; };
    
  struct TestAppArgs{
    std::string project;
    std::string inputPath;
    std::string inputName;
    std::string outputPath;
    std::string testname;
    std::string testArgs;
    std::string prototxt; // This should be folded into testArgs
    std::string caffemodel; // This should be folded into testArgs
    std::string cachemodel; // This should be folded into testArgs

    std::string profileName; // ok here?
    std::string profileFile;
    std::string configtarget;
    std::string calibTable;
    nvdla::QuantizationMode quantizationMode;

    NvU16 numBatches;
    nvdla::DataFormat inDataFormat;
    nvdla::DataType computePrecision;

    std::map<std::string, NvF32> tensorScales; };
  ```
  
2. Set up parameters

3. launchTest（)
  * testSetup
    1. clear wisdom file if any exist 
    2. Initiaize TestInfor
  * parseAndCompile
    1. Create new wisdom
    2. Parse
      * important classes
      ```c++
      class INetwork{
      public:
        virtual ITensor* addInput(const char * name, Dims4 dimensions) = 0;
        //	virtual void markChanged(const ILayer *) = 0;
        virtual bool markInput(ITensor * tensor) = 0;
        virtual void markOutput(ITensor * tensor) = 0;
        virtual IConvolutionLayer *    addConvolution   (ITensor * input, int numOutputs, int paddingValue, Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation, Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups) = 0;
        virtual IFullyConnectedLayer * addFullyConnected(ITensor * input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode) = 0;
        virtual IActivationLayer *     addActivation    (ITensor * input, ActivationType type) = 0;
        virtual IPoolingLayer *        addPooling       (ITensor * input, PoolingType type, Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding) = 0;
        virtual ILRNLayer *            addLRN           (ITensor * input, int window, float alpha, float beta, float k) = 0;
        virtual IScaleLayer *          addScale         (ITensor * input, ScaleMode mode, Weights shift, Weights scale, Weights power) = 0;
        virtual IBatchNormLayer *      addBatchNorm     (ITensor * input, BatchNormMode mode, Weights mean, Weights variance, float epsilon) = 0;
        virtual ISoftMaxLayer *        addSoftMax       (ITensor*input) = 0;
        virtual IConcatenationLayer *  addConcatenation (ITensor*const*inputs, int numInputs) = 0;
        virtual ISliceLayer *          addSlice         (ITensor*input, int numOutputs) = 0;
        virtual IDeconvolutionLayer *  addDeconvolution (ITensor * input, int numOutputs, int paddingValue, Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation, Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups) = 0;
        virtual IElementWiseLayer   *  addElementWise   (ITensor *input0, ITensor* input1, ElementWiseOperation op) = 0;

        virtual int getNumInputs()  const  = 0;
        virtual int getNumOutputs() const  = 0;
        virtual int getNumLayers()  const  = 0;

        virtual ILayer  * getLayer(int index)  const = 0;
        virtual ITensor * getOutput(int index) const = 0;
        virtual ITensor * getInput(int index)  const = 0;

        class OutputDimensionsFormula{
        public:
          virtual Dims2 compute(Dims2 inputDims, Dims2 kernelSize,  Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char* layerName) const = 0;
          virtual Dims2 compute(Dims2 inputDims, Dims2 kernelSize,  Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char* layerName) const = 0;
          virtual ~OutputDimensionsFormula() { }
        };

        class NetworkDefaultConvolutionFormula : public OutputDimensionsFormula{
        public:
          virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
          virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const;
        };

        class NetworkDefaultDeconvolutionFormula : public OutputDimensionsFormula{
        public:
          virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
          virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, Dims2 dilation, const char*) const;
        };

        class NetworkDefaultPoolingFormula : public OutputDimensionsFormula{
        public:
          virtual Dims2 compute(Dims2 input, Dims2 kernel, Dims2 stride, Dims2 tlPadding, Dims2 brPadding, const char*) const;
          virtual Dims2 compute(Dims2 /*input*/, Dims2 /*kernel*/, Dims2 /*stride*/, Dims2 /*tlPadding*/, Dims2 /*brPadding*/, Dims2 /*dilation*/, const char*) const{
            return Dims2(-1, -1);
          }
        };

        virtual void setPoolingOutputDimensionsFormula      (OutputDimensionsFormula* callback) = 0;
        virtual void setConvolutionOutputDimensionsFormula  (OutputDimensionsFormula* callback) = 0;
        virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback) = 0;

        virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula()       const = 0;
        virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula()   const = 0;
        virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const = 0;

        virtual const std::vector<ITensor *> & getInputs()  const = 0;
        virtual const std::vector<ILayer * > & getLayers()  const = 0;
        virtual const std::vector<ITensor *> & getOutputs() const = 0;

      protected:
      INetwork();
      virtual ~INetwork();
      };
      
      class Network : public INetwork{
      public: // externally facing
        virtual ITensor* addInput(const char* name, Dims4 dimensions);
        //	virtual void markChanged(const ILayer*);
        virtual bool markInput(ITensor * tensor);
        virtual void markOutput(ITensor* tensor);
        virtual IConvolutionLayer *    addConvolution(ITensor* input, int numOutputs, int paddingValue, Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation, Weights kernelWeights, Weights biasWeights, BiasMode biasmode, int numGroups);
        virtual IFullyConnectedLayer * addFullyConnected(ITensor* input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
        virtual IActivationLayer *     addActivation(ITensor* input, ActivationType type);
        virtual IPoolingLayer *        addPooling(ITensor* input, PoolingType type, Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding);
        virtual ILRNLayer *            addLRN(ITensor* input, int window, float alpha, float beta, float k);
        virtual IScaleLayer *          addScale(ITensor* input, ScaleMode mode, Weights shift, Weights scale, Weights power);
        virtual IBatchNormLayer *      addBatchNorm(ITensor* input, BatchNormMode mode, Weights mean, Weights variance, float epsilon);
        virtual ISoftMaxLayer *        addSoftMax(ITensor* input);
        virtual IConcatenationLayer *  addConcatenation(ITensor * const * inputs, int numInputs);
        virtual ISliceLayer *          addSlice(ITensor* input, int numOutputs);
        virtual IDeconvolutionLayer *  addDeconvolution(ITensor* input, int numOutputs, int paddingValue, Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation, Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
        virtual IElementWiseLayer *    addElementWise(ITensor* input0, ITensor* input1, ElementWiseOperation op);

        virtual int getNumInputs() const;
        virtual int getNumOutputs() const;
        virtual int getNumLayers() const ;

        virtual ILayer  * getLayer(int index)  const;
        virtual ITensor * getOutput(int index) const;
        virtual ITensor * getInput(int index)  const;

        virtual void setPoolingOutputDimensionsFormula      (OutputDimensionsFormula* callback);
        virtual void setConvolutionOutputDimensionsFormula  (OutputDimensionsFormula* callback);
        virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback);

        virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula()       const;
        virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula()   const;
        virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const;

        virtual const std::vector<ITensor *>& getInputs()  const;
        virtual const std::vector<ILayer * >& getLayers()  const;
        virtual const std::vector<ITensor *>& getOutputs() const;

        virtual NvU16 getFactoryType() const;

      public: // internally facing
        Network();
        virtual ~Network();
        virtual bool serializeTo(WisdomContainerEntry *) const;
        virtual bool deserializeFrom(WisdomContainerEntry *);
        virtual bool assignSymbols(Wisdom *);

      protected:
        friend class Wisdom;
        friend class NetworkFactory;
        void destroy();

      private:
        std::string newLayerName() const;
        std::string newTensorName() const;

        ITensor* addTensor(const std::string & s);
        const ILayer* findLayer(const std::string& name) const;
        bool checkNames(const char* name);

        std::vector<ITensor *> mTensors;
        std::vector<ILayer *>  mLayers;
        std::vector<ITensor *> mInputs;
        std::vector<ITensor *> mOutputs;

        // provides layer dimension caching. Layers can be mutated in any order and dimensions queried at any point.
        // So mutating a layer trims this, and querying always refills the cache up to the queried layer
        //	mutable std::vector<Dims3> mDimensions;

        // internal flags used by the builder that are not accessible through the API
        // int mInternalBuildFlags{ InternalBuildFlags::kENABLE_GRAPH_OPTIMIZATIONS };
        OutputDimensionsFormula* mConvDims, *mDeconvDims, *mPoolDims;
      };
      
      class IBlobNameToTensor{
      public:
        virtual void add(const std::string& name, ITensor* tensor) = 0;
        virtual ITensor* find(const char* name) const = 0;
        virtual ITensor*& operator[](const std::string& name) = 0;
        virtual void setTensorNames() = 0;
        virtual ~IBlobNameToTensor();
      };
      
      class BlobNameToTensor : public IBlobNameToTensor{
      public:
        virtual void add(const std::string& name, ITensor* tensor);
        virtual ITensor* find(const char* name) const;
        virtual ITensor*& operator[](const std::string& name);
        virtual void setTensorNames();
        virtual ~BlobNameToTensor();
      private:
        std::map<std::string, ITensor*> mMap;
      };

      ```
      * workflow
        1. parsing Caffe Network
        ```c++
        const IBlobNameToTensor* CaffeParser::parse(const char* deployFile, const char* modelFile, INetwork * network){
          ...
          network->setPoolingOutputDimensionsFormula(new CaffeParserPoolingDimsCallback);
          mModel = new dc::NetParameter();
          readBinaryProto(mModel/*.get()*/, modelFile, mProtobufBufferSize);
          mDeploy = new dc::NetParameter();
          readTextProto(mDeploy/*.get()*/, deployFile);
          CaffeWeightFactory weights(*mModel/**mModel.get()*/, false /*weightType == DataType::kHALF*/, mTmpAllocs);
          for (int i = 0; i < mDeploy->input_size(); i++){
            Dims4 dims;
            ... // setting dims parameter
            ITensor* tensor = network->addInput(mDeploy->input().Get(0).c_str(), dims);
            mBlobNameToTensor->add(mDeploy->input().Get(0), tensor);   // recording tensor info into mMap
          }
          for (int i = 0; i < mDeploy->layer_size() && ok; i++){
            const dc::LayerParameter& layerMsg = mDeploy->layer(i);
            if (layerMsg.type() == "Dropout"){
               mBlobNameToTensor->add(layerMsg.top().Get(0), mBlobNameToTensor->find(layerMsg.bottom().Get(0).c_str()));
               continue;
            }
            if (layerMsg.type() == "Input"){
              const dc::InputParameter& p = layerMsg.input_param();
              for (int i = 0; i < layerMsg.top_size(); i++){
                const dc::BlobShape& shape = p.shape().Get(i);
                Dims4 dims(shape.dim().Get(0), shape.dim().Get(1), shape.dim().Get(2), shape.dim().Get(3));
                ITensor* tensor = network->addInput(layerMsg.top(i).c_str(), dims);
                mBlobNameToTensor->add(layerMsg.top().Get(i), tensor);
              }
              continue;
            }
            if (layerMsg.type() == "Flatten"){
              ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
              (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
              std::cout << "Warning: Flatten layer ignored." << std::endl;
              continue;
            }
            LayerParseFnMap::iterator v = gParseTable.find(layerMsg.type());
            ILayer* layer = (*v->second)(network, layerMsg, weights, mBlobNameToTensor);
            layer->setName(layerMsg.name().c_str());
            mBlobNameToTensor->add(layerMsg.top(0), layer->getOutput(0));
          }
        }
        ```
        2. marking the network's outputs
        3. parsing and setting tensor scales according to computation precision
        4. attaching parsed network to the wisdom
        ```c++
        wisdom->setNetworkTransient(network);
        ```
       
  3. Compile


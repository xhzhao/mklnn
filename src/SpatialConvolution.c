#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SpatialConvolution.c"
#else

dnnError_t  MKLNN_(init_conversion)(
  dnnPrimitive_t *cvt,
  real **ptr_out,
  dnnLayout_t lt_pr,
  dnnLayout_t lt_us)
{
  dnnError_t err;
  *ptr_out = NULL;
  if(!MKLDNN_(dnnLayoutCompare)(lt_pr, lt_us)) {
    CHECK_ERR( MKLDNN_(dnnConversionCreate)(cvt, lt_us, lt_pr), err );
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)ptr_out, lt_pr), err );
  }
  return E_SUCCESS;
}

static void MKLNN_(SpatialConvolution_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *output,
  int outC,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group)
{
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];
  int outH = (inH + 2*padH - kH)/dH + 1;
  int outW = (inW + 2*padW - kW)/dW + 1;

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN__init_forward: start.");
  fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif

  dnnError_t err;
  dnnPrimitive_t conv_forward = NULL;
  dnnPrimitive_t conv_bwd_data = NULL;
  dnnPrimitive_t conv_bwd_filter = NULL;
  dnnPrimitive_t conv_bwd_bias = NULL;

  int    f_DIMENSION           = DIMENSION + (group != 1);
  size_t inputSize[DIMENSION]  = {inW, inH, inC, N};
  size_t filterSize[5]         = {kW, kH, inC/group, outC/group, group};
  size_t outputSize[DIMENSION] = {outW, outH, outC, N};
  size_t stride[DIMENSION-2]   = {dW, dH};
  int    pad[DIMENSION-2]      = {-padW, -padH};

  size_t outputStrides[DIMENSION] = {1, outW, outH*outW, outC*outH*outW};
  size_t inputStrides[DIMENSION]  = {1, inW, inH*inW, inC*inH*inW};
  size_t filterStrides[5]         = {1, kW, kH*kW, (inC/group)*kH*kW, (inC/group)*(outC/group)*kH*kW};

  size_t biasSize[1]    = {outputSize[2]};
  size_t biasStrides[1] = {1};

  dnnLayout_t lt_user_input  = NULL;
  dnnLayout_t lt_user_filter = NULL;
  dnnLayout_t lt_user_bias   = NULL;
  dnnLayout_t lt_user_output = NULL;

  dnnLayout_t lt_forward_conv_input = NULL;
  dnnLayout_t lt_forward_conv_filter = NULL;
  dnnLayout_t lt_forward_conv_bias = NULL;
  dnnLayout_t lt_forward_conv_output = NULL;

  //forward conversions and buffers
  dnnPrimitive_t cvt_forward_input = NULL;
  dnnPrimitive_t cvt_forward_filter = NULL;
  dnnPrimitive_t cvt_forward_bias = NULL;
  dnnPrimitive_t cvt_forward_output_back = NULL;

  real *buffer_forward_input = NULL;
  real *buffer_forward_filter = NULL;
  real *buffer_forward_bias = NULL;
  real *buffer_forward_output = NULL;

  dnnPrimitiveAttributes_t attributes = NULL;
  int input_layout_create_local = 0;

  CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesCreate)(&attributes), err );

  if(0 == input->workspace || (0 == input->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_user_input = input->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_filter, f_DIMENSION, filterSize, filterStrides), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_bias, 1, biasSize, biasStrides), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_output, DIMENSION, outputSize, outputStrides), err );

  CHECK_ERR( MKLDNN_(dnnGroupsConvolutionCreateForwardBias)(&conv_forward, attributes, dnnAlgorithmConvolutionDirect, group, DIMENSION, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  CHECK_ERR( MKLDNN_(dnnGroupsConvolutionCreateBackwardData)(&conv_bwd_data, attributes, dnnAlgorithmConvolutionDirect, group, DIMENSION, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  CHECK_ERR( MKLDNN_(dnnGroupsConvolutionCreateBackwardFilter)(&conv_bwd_filter, attributes, dnnAlgorithmConvolutionDirect, group, DIMENSION, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  CHECK_ERR( MKLDNN_(dnnGroupsConvolutionCreateBackwardBias)(&conv_bwd_bias, attributes, dnnAlgorithmConvolutionDirect, group, DIMENSION, outputSize),err);

  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_forward_conv_input, conv_forward, dnnResourceSrc), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_forward_conv_filter, conv_forward, dnnResourceFilter), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_forward_conv_bias, conv_forward, dnnResourceBias), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_forward_conv_output, conv_forward, dnnResourceDst), err );

  //init forward conversions:
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_forward_input, &buffer_forward_input, lt_forward_conv_input, lt_user_input), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_forward_filter, &buffer_forward_filter, lt_forward_conv_filter, lt_user_filter), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_forward_bias, &buffer_forward_bias, lt_forward_conv_bias, lt_user_bias), err );

#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_forward_conv_output);
  int size2 = MKLDNN_(dnnLayoutGetMemorySize)(lt_user_output);
  if(size1 == size2 && size2 == (outW*outH*outC*N*sizeof(real))) {
#if 0
    fprintf(stderr, "MKLDNN_ Convolution forward ouput layout match OK\n");
#endif
  } else {
    if(!MKLDNN_(dnnLayoutCompare)(lt_forward_conv_output, lt_user_output)) {
      CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_forward_output_back, lt_user_output, lt_forward_conv_output), err );
    }
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_forward_output), lt_forward_conv_output), err );
  }
#endif

  if(attributes) { 
    CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesDestroy)(&attributes), err );
  }
  if(input_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_input), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_filter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_bias), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_output), err);
  
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_forward_conv_input), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_forward_conv_filter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_forward_conv_bias), err);

  TH_MKL_(resize4d)(output, N, outC, outH, outW);

  dnnWorkspace* outputWorkspace = WORKSPACE_(New)(lt_forward_conv_output);
  if(cvt_forward_output_back) {
    outputWorkspace->cvtPrmt = cvt_forward_output_back;
    outputWorkspace->sync = 1;
  }
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  //save the output layout to dnnPrimitive
  primitives->tensor->storage->data[CONV_PRIM_FWD]               = (long)conv_forward;
  primitives->tensor->storage->data[CONV_PRIM_BWD_DATA]          = (long)conv_bwd_data;
  primitives->tensor->storage->data[CONV_PRIM_BWD_FILTER]        = (long)conv_bwd_filter;
  primitives->tensor->storage->data[CONV_PRIM_BWD_BIAS]          = (long)conv_bwd_bias;

  primitives->tensor->storage->data[CONV_PRIM_CVT_INPUT_U2F]     = (long)cvt_forward_input;
  primitives->tensor->storage->data[CONV_PRIM_CVT_FILTER_U2F]    = (long)cvt_forward_filter;
  primitives->tensor->storage->data[CONV_PRIM_CVT_BIAS_U2F]      = (long)cvt_forward_bias;

  primitives->tensor->storage->data[CONV_BUFFER_INPUT_FWD]       = (long)buffer_forward_input;
  primitives->tensor->storage->data[CONV_BUFFER_FILTER_FWD]      = (long)buffer_forward_filter;
  primitives->tensor->storage->data[CONV_BUFFER_BIAS_FWD]        = (long)buffer_forward_bias;
  primitives->tensor->storage->data[CONV_BUFFER_OUTPUT_FWD]      = (long)buffer_forward_output;

#if LOG_ENABLE
  printf("cvt_forward_input=%d,cvt_forward_filter=%d,cvt_forward_output_back=%d \n", cvt_forward_input,cvt_forward_filter, cvt_forward_output_back);
  printf("SpatialConvolutionMM_MKLDNN__init_forward: end, sizeof(real)=%d\n",sizeof(real));
#endif

}

static void MKLNN_(SpatialConvolution_init_bwddata)(
  THMKLLongTensor *primitives,
  THMKLTensor* input,
  THMKLTensor* gradOutput,
  THMKLTensor* gradInput,
  int kH,
  int kW,
  int dH,
  int dW,
  int padH,
  int padW,
  int group)
{
  TH_MKL_(resizeAs)(gradInput, input);

  int N = gradInput->size[0];
  int gradInC = gradInput->size[1];
  int gradInH = gradInput->size[2];
  int gradInW = gradInput->size[3];

  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN__init_bwddata: start.");
  fprintf(stderr, "N=%d, gradInC=%d, gradInH=%d, gradInW=%d, kH=%d, kW=%d, dH=%d, dW=%d, padH=%d, padW=%d, gradOutC=%d, gradOutH=%d, gradOutW=%d\n", N, gradInC, gradInH, gradInW, kH, kW, dH, dW, padH, padW, gradOutC, gradOutH, gradOutW);
#endif

  dnnError_t err;
  dnnPrimitive_t conv_bwd_data = NULL;

  int f_DIMENSION = DIMENSION + (group != 1);
  size_t gradInputSize[DIMENSION]  = {gradInW, gradInH, gradInC, N};
  size_t gradOutputSize[DIMENSION] = {gradOutW, gradOutH, gradOutC, N};
  size_t filterSize[5] = {kW, kH, gradInC/group, gradOutC/group, group};

  size_t stride[DIMENSION-2]   = {dW,dH};
  int    pad[DIMENSION-2]      = {-padW,-padH};

  size_t gradOutputStrides[DIMENSION] = {1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW};
  size_t gradInputStrides[DIMENSION]  = {1, gradInW, gradInH*gradInW, gradInC*gradInH*gradInW};
  size_t filterStrides[5]         = {1, kW, kH*kW, (gradInC/group)*kH*kW, (gradInC/group)*(gradOutC/group)*kH*kW};
  size_t biasSize[1]              = {gradOutputSize[2] };
  size_t biasStrides[1]           = {1};

  dnnLayout_t lt_user_gradInput = NULL;
  dnnLayout_t lt_user_filter = NULL;
  dnnLayout_t lt_user_gradOutput = NULL;
  dnnLayout_t lt_bwddata_conv_gradInput = NULL;
  dnnLayout_t lt_bwddata_conv_filter = NULL;
  dnnLayout_t lt_bwddata_conv_gradOutput = NULL; 

  dnnPrimitive_t cvt_bwddata_gradInput_back = NULL;
  dnnPrimitive_t cvt_bwddata_filter = NULL;
  dnnPrimitive_t cvt_bwddata_gradOutput = NULL;

  real *buffer_bwddata_gradInput  = NULL;
  real *buffer_bwddata_filter = NULL;
  real *buffer_bwddata_gradOutput = NULL;
  int gradOutput_layout_create_local = 0;

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradInput, DIMENSION, gradInputSize, gradInputStrides), err );

  conv_bwd_data = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_DATA]);
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwddata_conv_gradInput, conv_bwd_data, dnnResourceDiffSrc), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwddata_conv_filter, conv_bwd_data, dnnResourceFilter), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwddata_conv_gradOutput, conv_bwd_data, dnnResourceDiffDst), err );

  //get forward filter layout, convert from forward filter to bdwdata filter
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_filter, f_DIMENSION, filterSize, filterStrides), err );

  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwddata_filter, &buffer_bwddata_filter, lt_bwddata_conv_filter, lt_user_filter), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwddata_gradOutput, &buffer_bwddata_gradOutput, lt_bwddata_conv_gradOutput, lt_user_gradOutput), err );

#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_bwddata_conv_gradInput);
  int size2 = MKLDNN_(dnnLayoutGetMemorySize)(lt_user_gradInput);
  if(size1 == size2 && size2 == (gradInW*gradInH*gradInC*N*sizeof(real))) {
#if 0
    fprintf(stderr,"MKLDNN_ Convolution bwddata input layout match OK\n");
#endif
  } else {
    if(!MKLDNN_(dnnLayoutCompare)(lt_bwddata_conv_gradInput, lt_user_gradInput)) {
      CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_bwddata_gradInput_back, lt_user_gradInput, lt_bwddata_conv_gradInput), err );
    }
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_bwddata_gradInput), lt_bwddata_conv_gradInput), err );
  }
#endif

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradInput), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_filter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwddata_conv_filter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwddata_conv_gradOutput), err);

  dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_bwddata_conv_gradInput);
  if(cvt_bwddata_gradInput_back) {
    gradInputWorkspace->cvtPrmt = cvt_bwddata_gradInput_back;
    gradInputWorkspace->sync = 1;
  }

  TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);

  //save the output layout to dnnPrimitive
  primitives->tensor->storage->data[CONV_PRIM_CVT_FILTER_U2BD]            = (long)cvt_bwddata_filter;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BD]        = (long)cvt_bwddata_gradOutput;

  primitives->tensor->storage->data[CONV_BUFFER_GRADINPUT_BWD_DATA]       = (long)buffer_bwddata_gradInput;
  primitives->tensor->storage->data[CONV_BUFFER_FILTER_BWD_DATA]          = (long)buffer_bwddata_filter;
  primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_DATA]      = (long)buffer_bwddata_gradOutput;

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN__init_bwddata: end, sizeof(real)=%d\n",sizeof(real));
#endif

}

static void MKLNN_(SpatialConvolution_init_bwdfilter)(
  THMKLLongTensor *primitives,
  THMKLTensor* input,
  THMKLTensor* gradOutput,
  int kH,
  int kW,
  int dH,
  int dW,
  int padH,
  int padW,
  int group)
{
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN__init_bwdfilter: start.");
  fprintf(stderr, "N=%d, inC=%d, inH=%d, inW=%d, kH=%d, kW=%d, dH=%d, dW=%d, padH=%d, padW=%d, gradOutC=%d, gradOutH=%d, gradOutW=%d\n", N, inC, inH, inW, kH, kW, dH, dW, padH, padW, gradOutC, gradOutH, gradOutW);
#endif

  dnnError_t err;
  dnnPrimitive_t conv_bwd_filter = NULL;
  dnnPrimitive_t conv_bwd_bias = NULL;


  dnnLayout_t lt_user_input  = NULL;
  dnnLayout_t lt_user_gradFilter = NULL;
  dnnLayout_t lt_user_gradBias   = NULL;
  dnnLayout_t lt_user_gradOutput = NULL;
  dnnLayout_t lt_bwdfilter_conv_input = NULL;
  dnnLayout_t lt_bwdfilter_conv_gradFilter = NULL;
  dnnLayout_t lt_bwdfilter_conv_gradOutput = NULL;

  //backward filter conversions and buffers
  dnnPrimitive_t cvt_bwdfilter_input = NULL;
  dnnPrimitive_t cvt_bwdfilter_gradFilter = NULL;
  dnnPrimitive_t cvt_bwdfilter_gradFilter_back = NULL;
  dnnPrimitive_t cvt_bwdfilter_gradOutput = NULL;

  real *buffer_bwdfilter_input  = NULL;
  real *buffer_bwdfilter_gradFilter = NULL;
  real *buffer_bwdfilter_gradOutput = NULL;

  dnnLayout_t lt_bwdbias_conv_gradBias = NULL;
  dnnLayout_t lt_bwdbias_conv_gradOutput = NULL;

  dnnPrimitive_t cvt_bwdbias_gradBias = NULL;
  dnnPrimitive_t cvt_bwdbias_gradOutput = NULL;
  dnnPrimitive_t cvt_bwdbias_gradBias_back = NULL;

  real *buffer_bwdbias_gradBias = NULL;
  real *buffer_bwdbias_gradOutput = NULL;

  int input_layout_create_local = 0;
  int gradOutput_layout_create_local = 0;

  int    f_DIMENSION           = DIMENSION + (group != 1);
  size_t inputSize[DIMENSION]  = {inW, inH, inC, N};
  size_t gradFilterSize[5]     = {kW, kH, inC/group, gradOutC/group, group};
  size_t gradOutputSize[DIMENSION] = {gradOutW, gradOutH, gradOutC, N};

  size_t stride[DIMENSION-2]   = {dW, dH};
  int    pad[DIMENSION-2]      = {-padW, -padH};

  size_t gradOutputStrides[DIMENSION] = {1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW};
  size_t inputStrides[DIMENSION]  = {1, inW, inH*inW, inC*inH*inW};
  size_t gradFilterStrides[5]     = {1, kW, kH*kW, (inC/group)*kH*kW, (inC/group)*(gradOutC/group)*kH*kW};

  size_t gradBiasSize[1]    = {gradOutputSize[2]};
  size_t gradBiasStrides[1] = {1};

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }

  if((0 == input->workspace) || (0 == input->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_user_input = input->workspace->layout;
  }
  
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradFilter, f_DIMENSION, gradFilterSize, gradFilterStrides), err );
  conv_bwd_filter = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_FILTER]);

  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwdfilter_conv_input,      conv_bwd_filter, dnnResourceSrc), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwdfilter_conv_gradFilter, conv_bwd_filter, dnnResourceDiffFilter), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwdfilter_conv_gradOutput, conv_bwd_filter, dnnResourceDiffDst), err );

  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwdfilter_input,      &buffer_bwdfilter_input,      lt_bwdfilter_conv_input,      lt_user_input), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwdfilter_gradOutput, &buffer_bwdfilter_gradOutput, lt_bwdfilter_conv_gradOutput, lt_user_gradOutput), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwdfilter_gradFilter, &buffer_bwdfilter_gradFilter, lt_bwdfilter_conv_gradFilter, lt_user_gradFilter), err );

  if(!MKLDNN_(dnnLayoutCompare)(lt_user_gradFilter, lt_bwdfilter_conv_gradFilter)) {
    CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_bwdfilter_gradFilter_back, lt_bwdfilter_conv_gradFilter, lt_user_gradFilter), err );
  }

  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradBias, 1, gradBiasSize, gradBiasStrides), err );
  conv_bwd_bias = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_BIAS]);

  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwdbias_conv_gradBias, conv_bwd_bias, dnnResourceDiffBias), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bwdbias_conv_gradOutput, conv_bwd_bias, dnnResourceDiffDst), err );

  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwdbias_gradOutput, &buffer_bwdbias_gradOutput, lt_bwdbias_conv_gradOutput, lt_user_gradOutput), err );
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_bwdbias_gradBias, &buffer_bwdbias_gradBias, lt_bwdbias_conv_gradBias, lt_user_gradBias), err );
  if(!MKLDNN_(dnnLayoutCompare)(lt_user_gradBias, lt_bwdbias_conv_gradBias)) {
    CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_bwdbias_gradBias_back, lt_bwdbias_conv_gradBias, lt_user_gradBias), err );
  }

  if(input_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_input), err);
  }

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }

  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradFilter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwdfilter_conv_input), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwdfilter_conv_gradFilter), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwdfilter_conv_gradOutput), err);

  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradBias), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwdbias_conv_gradBias), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bwdbias_conv_gradOutput), err);

  //save the dnnPrimitive to THTensor(long int array)
  primitives->tensor->storage->data[CONV_PRIM_CVT_INPUT_U2BF]      = (long)cvt_bwdfilter_input;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADFILTER_U2BF] = (long)cvt_bwdfilter_gradFilter;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BF] = (long)cvt_bwdfilter_gradOutput;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADFILTER_BF2U] = (long)cvt_bwdfilter_gradFilter_back;

  primitives->tensor->storage->data[CONV_BUFFER_INPUT_BWD_FILTER]      = (long)buffer_bwdfilter_input;
  primitives->tensor->storage->data[CONV_BUFFER_GRADFILTER_BWD_FILTER] = (long)buffer_bwdfilter_gradFilter;
  primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_FILTER] = (long)buffer_bwdfilter_gradOutput;

  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADBIAS_U2BB] = (long)cvt_bwdbias_gradBias;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADBIAS_BB2U] = (long)cvt_bwdbias_gradBias_back;
  primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BB] = (long)cvt_bwdbias_gradOutput;

  primitives->tensor->storage->data[CONV_BUFFER_GRADBIAS_BWD_BIAS] = (long)buffer_bwdbias_gradBias;
  primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_BIAS] = (long)buffer_bwdbias_gradOutput;

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN__init_bwdfilter: end, sizeof(real)=%d\n",sizeof(real));
#endif

}


void MKLNN_(SpatialConvolution_forward)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *output,
  THTensor *weight,
  THTensor *bias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group)
{
  struct timeval start, init, convert, execute, end;
  gettimeofday(&start, NULL);

  if(initOK == 0) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT]   = MKL_CONV_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_CONV_BUFFER;
    int outC = weight->size[0];
    MKLNN_(SpatialConvolution_init_forward)(primitives, input, output, outC, kH, kW, dH, dW, padH, padW, group);
  }
  gettimeofday(&init, NULL);

  dnnError_t err;
  dnnPrimitive_t conv_forward = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_FWD];
  dnnPrimitive_t cvt_forward_input = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_INPUT_U2F];
  dnnPrimitive_t cvt_forward_filter = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_FILTER_U2F]; 
  dnnPrimitive_t cvt_forward_bias = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_BIAS_U2F];

  real *buffer_forward_input = (real *)(primitives->tensor->storage->data[CONV_BUFFER_INPUT_FWD]);
  real *buffer_forward_filter = (real *)(primitives->tensor->storage->data[CONV_BUFFER_FILTER_FWD]);
  real *buffer_forward_bias = (real *)(primitives->tensor->storage->data[CONV_BUFFER_BIAS_FWD]);
  real *buffer_forward_output = (real *)(primitives->tensor->storage->data[CONV_BUFFER_OUTPUT_FWD]);

  real *inPtr     = TH_MKL_(data)(input);
  real *outPtr    = TH_MKL_(data)(output);
  real *filterPtr = THTensor_(data)(weight);
  real *biasPtr   = THTensor_(data)(bias);

  //The ouput data is modified by the computation
  if(NULL != buffer_forward_output) {
    fprintf(stderr, "%s Fatal error\n", __func__);
    output->dnnMem = 1;
    outPtr = buffer_forward_output;
  }

  real *resConv[dnnResourceNumber] = {0};
  resConv[dnnResourceSrc]    = inPtr;
  resConv[dnnResourceFilter] = filterPtr;
  resConv[dnnResourceBias]   = biasPtr;
  resConv[dnnResourceDst]    = outPtr;

  if(cvt_forward_input) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_forward_input, inPtr, buffer_forward_input), err );
    resConv[dnnResourceSrc] = buffer_forward_input;
  }

  if(cvt_forward_filter) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_forward_filter, filterPtr, buffer_forward_filter), err );
    resConv[dnnResourceFilter] = buffer_forward_filter;
  }

  if(cvt_forward_bias) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_forward_bias, biasPtr, buffer_forward_bias), err );
    resConv[dnnResourceBias] = buffer_forward_bias;
  }

  gettimeofday(&convert, NULL);

  CHECK_ERR(MKLDNN_(dnnExecute)(conv_forward, (void**)resConv),err);
  gettimeofday(&execute, NULL);


  gettimeofday(&end, NULL);
#if LOG_ENABLE
  double init_t    = (init.tv_sec - start.tv_sec) * 1000 + (double)(init.tv_usec - start.tv_usec) /1000;
  double convert_t = (convert.tv_sec - init.tv_sec) * 1000 + (double)(convert.tv_usec - init.tv_usec) /1000;
  double exec_t    = (execute.tv_sec - convert.tv_sec) * 1000 + (double)(execute.tv_usec - convert.tv_usec) /1000;
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv forward init_time = %.2f ms, convert_time = %.2f, exec_time = %.2f, all_time=%.2f ms\n", init_t, convert_t, exec_t, all_t);
#endif

#if MKL_TIME
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv forward all_time = %.2f ms \n", all_t);
#endif
}


void MKLNN_(SpatialConvolution_bwdData)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput,
  THTensor *weight,
  THTensor *bias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group )
{
  struct timeval start, init, convert, execute, end;

  gettimeofday(&start, NULL);
  if(0 == initOK) {
    MKLNN_(SpatialConvolution_init_bwddata)(primitives, input, gradOutput, gradInput, kH, kW, dH, dW, padH, padW, group);
  }
  gettimeofday(&init, NULL);

  dnnError_t err;
  dnnPrimitive_t conv_bwdData = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_DATA]);
  dnnPrimitive_t cvt_bwddata_filter = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_FILTER_U2BD];
  dnnPrimitive_t cvt_bwddata_gradOutput = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BD];

  real *buffer_bwddata_gradInput = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADINPUT_BWD_DATA]);
  real *buffer_bwddata_filter = (real *)(primitives->tensor->storage->data[CONV_BUFFER_FILTER_BWD_DATA]);
  real *buffer_bwddata_gradOutput = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_DATA]);

  real *gradInPtr = TH_MKL_(data)(gradInput);
  real *filterPtr = THTensor_(data)(weight);
  real *gradOutPtr = TH_MKL_(data)(gradOutput);

  if(buffer_bwddata_gradInput){
    fprintf(stderr, "%s Fatal error\n", __func__);
    gradInput->dnnMem = 1; 
    gradInPtr = buffer_bwddata_gradInput;
  }

  real *resConv[dnnResourceNumber]= {0};
  resConv[dnnResourceFilter] = filterPtr;
  resConv[dnnResourceDiffDst] = gradOutPtr;
  resConv[dnnResourceDiffSrc] = gradInPtr;

  if(cvt_bwddata_gradOutput) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwddata_gradOutput, gradOutPtr, buffer_bwddata_gradOutput), err );
    resConv[dnnResourceDiffDst] = buffer_bwddata_gradOutput;
  }

  if(cvt_bwddata_filter) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwddata_filter, filterPtr, buffer_bwddata_filter), err );
    resConv[dnnResourceFilter] = buffer_bwddata_filter;
  }
  gettimeofday(&convert, NULL);

  CHECK_ERR(MKLDNN_(dnnExecute)(conv_bwdData, (void**)resConv),err);
  gettimeofday(&execute, NULL);

  gettimeofday(&end,NULL);

#if LOG_ENABLE
  double init_t    = (init.tv_sec - start.tv_sec) * 1000 + (double)(init.tv_usec - start.tv_usec) /1000;
  double convert_t = (convert.tv_sec - init.tv_sec) * 1000 + (double)(convert.tv_usec - init.tv_usec) /1000;
  double exec_t    = (execute.tv_sec - convert.tv_sec) * 1000 + (double)(execute.tv_usec - convert.tv_usec) /1000;
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv backward dada init_time = %.2f ms, convert_time = %.2f, exec_time = %.2f, all_time=%.2f ms\n", init_t, convert_t, exec_t, all_t);
#endif

#if MKL_TIME
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv backward data time = %.2f ms \n", all_t);
#endif
}

void MKLNN_(SpatialConvolution_bwdFilter)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  real scale,
  int group)
{
  struct timeval start, init, convert1, execute1, convert2, execute2, end;
  gettimeofday(&start, NULL);
  if(0 == initOK) {
    MKLNN_(SpatialConvolution_init_bwdfilter)(primitives, input, gradOutput, kH, kW, dH, dW, padH, padW, group);
  }
  gettimeofday(&init, NULL);

  dnnError_t err;
  dnnPrimitive_t conv_bwdFilter = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_FILTER]);
  dnnPrimitive_t conv_bwdBias = (dnnPrimitive_t) (primitives->tensor->storage->data[CONV_PRIM_BWD_BIAS]);

  dnnPrimitive_t cvt_bwdfilter_input = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_INPUT_U2BF];
  dnnPrimitive_t cvt_bwdfilter_gradFilter = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADFILTER_U2BF];
  dnnPrimitive_t cvt_bwdfilter_gradOutput = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BF];
  dnnPrimitive_t cvt_bwdfilter_gradFilter_back = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADFILTER_BF2U];

  real *buffer_bwdfilter_input = (real *)(primitives->tensor->storage->data[CONV_BUFFER_INPUT_BWD_FILTER]);
  real *buffer_bwdfilter_gradFilter = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADFILTER_BWD_FILTER]);
  real *buffer_bwdfilter_gradOutput = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_FILTER]);

  dnnPrimitive_t cvt_bwdbias_gradBias = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADBIAS_U2BB];
  dnnPrimitive_t cvt_bwdbias_gradOutput = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADOUTPUT_U2BB];
  dnnPrimitive_t cvt_bwdbias_gradBias_back = (dnnPrimitive_t)primitives->tensor->storage->data[CONV_PRIM_CVT_GRADBIAS_BB2U];

  real *buffer_bwdbias_gradBias = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADBIAS_BWD_BIAS]);
  real *buffer_bwdbias_gradOutput = (real *)(primitives->tensor->storage->data[CONV_BUFFER_GRADOUTPUT_BWD_BIAS]);

  real *inPtr = TH_MKL_(data)(input);
  real *gradOutPtr = TH_MKL_(data)(gradOutput);
  real *gradFilterPtr = THTensor_(data)(gradWeight);
  real *gradBiasPtr = THTensor_(data)(gradBias);

  real *resConv[dnnResourceNumber] = {0};
  resConv[dnnResourceSrc] = inPtr;
  resConv[dnnResourceDiffFilter] = gradFilterPtr;
  resConv[dnnResourceDiffDst] = gradOutPtr;
  resConv[dnnResourceDiffBias] = gradBiasPtr;


  real *resBias[dnnResourceNumber]= {0};
  resBias[dnnResourceDiffDst] = gradOutPtr;
  resBias[dnnResourceDiffBias] = gradBiasPtr;

  if(cvt_bwdfilter_input) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwdfilter_input, inPtr, buffer_bwdfilter_input), err );
    resConv[dnnResourceSrc] = buffer_bwdfilter_input;
  }
  if(cvt_bwdfilter_gradOutput) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwdfilter_gradOutput, gradOutPtr, buffer_bwdfilter_gradOutput), err );
    resConv[dnnResourceDiffDst] = buffer_bwdfilter_gradOutput;
  }
  if(cvt_bwdfilter_gradFilter) {
    resConv[dnnResourceDiffFilter] = buffer_bwdfilter_gradFilter;
  }

  gettimeofday(&convert1,NULL);
  CHECK_ERR( MKLDNN_(dnnExecute)(conv_bwdFilter, (void**)resConv), err);

  if(cvt_bwdfilter_gradFilter_back) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwdfilter_gradFilter_back, buffer_bwdfilter_gradFilter, gradFilterPtr), err );
  }
  gettimeofday(&execute1, NULL);

  if(cvt_bwdbias_gradOutput) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwdbias_gradOutput, buffer_bwdbias_gradOutput, gradOutPtr), err );
    resBias[dnnResourceDiffDst] = buffer_bwdbias_gradOutput;
  }

  if(cvt_bwdbias_gradBias) {
    resBias[dnnResourceDiffBias] = buffer_bwdbias_gradBias;
  }

  gettimeofday(&convert2,NULL);
  CHECK_ERR(MKLDNN_(dnnExecute)(conv_bwdBias, (void**)resBias),err);

  if(cvt_bwdbias_gradBias_back) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_bwdbias_gradBias_back, buffer_bwdbias_gradBias, gradBiasPtr), err );
  }
  gettimeofday(&execute2, NULL);

  gettimeofday(&end,NULL);
#if LOG_ENABLE
  double init_t    = (init.tv_sec - start.tv_sec) * 1000 + (double)(init.tv_usec - start.tv_usec) /1000;
  double convert_t1 = (convert1.tv_sec - init.tv_sec) * 1000 + (double)(convert1.tv_usec - init.tv_usec) /1000;
  double exec_t1    = (execute1.tv_sec - convert.tv_sec) * 1000 + (double)(execute1.tv_usec - convert1.tv_usec) /1000;
  double convert_t2 = (convert2.tv_sec - init.tv_sec) * 1000 + (double)(convert2.tv_usec - execute1.tv_usec) /1000;
  double exec_t2    = (execute2.tv_sec - convert.tv_sec) * 1000 + (double)(execute2.tv_usec - convert2.tv_usec) /1000;
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv backward param init_time = %.2f ms, convert1_time = %.2f, exec1_time = %.2f, convert2_time = %.2f, exec2_time = %.2f, all_time=%.2f ms\n", init_t, convert_t1, exec_t1, convert_t2, exec_t2, all_t);
#endif

#if MKL_TIME
  double all_t     = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"    Conv backward data time = %.2f ms \n", all_t);
#endif

}
#endif

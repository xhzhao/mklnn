#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/Threshold.c"
#else

static void MKLNN_(SpatialConvolution_Relu_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *output,
  real threshold)
{
  TH_MKL_(resizeAs)(output, input);
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];
  size_t inputSize[DIMENSION]    = {inW, inH, inC, N};
  size_t inputStrides[DIMENSION] = {1, inW, inH*inW, inC*inH*inW};

  /*relu operation is independent of the data layout, and output layout coordinate with the input*/
  dnnError_t err;
  dnnPrimitive_t relu_forward = NULL;
  dnnPrimitive_t relu_backward = NULL;
  dnnLayout_t lt_relu_input = NULL;
  dnnPrimitiveAttributes_t attributes = NULL;
  real *buffer_forward_output = NULL;
  int input_layout_create_local = 0;

  CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesCreate)(&attributes), err );
  if((0 == input->workspace) || (0 == input->workspace->layout)) {
    CHECK_ERR(MKLDNN_(dnnLayoutCreate)(&lt_relu_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_relu_input = input->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnReLUCreateForward)(&relu_forward, attributes, lt_relu_input, threshold), err );
  CHECK_ERR( MKLDNN_(dnnReLUCreateBackward)(&relu_backward, attributes, lt_relu_input, lt_relu_input, threshold), err );

#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_relu_input);
  int size2 = inW*inH*inC*N*sizeof(real);
  if(size1 == size2) {
#if 0
    fprintf(stderr,"MKLDNN_ Relu forward ouput layout match OK\n");
#endif 
  } else {
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_forward_output), lt_relu_input), err );
  }
#endif
  if (attributes) {
    CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesDestroy)(&attributes), err ); 
  }
  
  dnnWorkspace* outputWorkspace = NULL;
  if(input_layout_create_local) {
    outputWorkspace = WORKSPACE_(New)(lt_relu_input);
  } else {
    outputWorkspace = input->workspace;
    WORKSPACE_(Retain)(input->workspace);
  }
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  primitives->tensor->storage->data[RELU_PRIM_FWD]   = (long)relu_forward;
  primitives->tensor->storage->data[RELU_PRIM_BWD]   = (long)relu_backward;
  primitives->tensor->storage->data[RELU_BUFFER_OUTPUT_FWD]  = (long)buffer_forward_output;
}

static void MKLNN_(SpatialConvolution_Relu_init_backward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput)
{
  TH_MKL_(resizeAs)(gradInput, input);

  int N = gradInput->size[0];
  int gradInC = gradInput->size[1];
  int gradInH = gradInput->size[2];
  int gradInW = gradInput->size[3];
  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];

  dnnError_t err;
  dnnPrimitive_t relu_backward = (dnnPrimitive_t) (primitives->tensor->storage->data[RELU_PRIM_BWD]);
  dnnLayout_t lt_relu_diff_dst = NULL;
  dnnLayout_t lt_relu_diff_src = NULL;
  dnnLayout_t lt_user_gradOutput = NULL;
  dnnLayout_t lt_user_gradInput = NULL;
  dnnPrimitive_t cvt_backward_gradOutput = NULL;
  dnnPrimitive_t cvt_backward_gradInput_back = NULL;
  real *buffer_backward_gradOutput = NULL;
  real *buffer_backward_gradInput = NULL;

  size_t gradOutputSize[DIMENSION] = {gradOutW, gradOutH, gradOutC, N};
  size_t gradInputSize[DIMENSION] = {gradInW, gradInH, gradInC, N};
  size_t gradOutputStrides[DIMENSION] = {1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW};
  size_t gradInputStrides[DIMENSION] = {1, gradInW, gradInH*gradInW, gradInC*gradInH*gradInW};
  int gradOutput_layout_create_local = 0;

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradInput, DIMENSION, gradInputSize, gradInputStrides), err );

  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_relu_diff_dst, relu_backward, dnnResourceDiffDst), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_relu_diff_src, relu_backward, dnnResourceDiffSrc), err );


#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_relu_diff_src);
  int size2 = gradInW*gradInH*gradInC*N*sizeof(real);
  if(size1 == size2) {
#if 0 
    fprintf(stderr,"MKLDNN_ Relu bwddata input layout match OK\n");
#endif
  } else {
    if(!MKLDNN_(dnnLayoutCompare)(lt_relu_diff_src, lt_user_gradInput)) {
      CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_backward_gradInput_back, lt_user_gradInput, lt_relu_diff_src), err );
    }
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_backward_gradInput), lt_relu_diff_src), err );
    fprintf(stderr,"MKLDNN_ Relu bwddata input layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
  }
#endif

  CHECK_ERR(MKLNN_(init_conversion)(&cvt_backward_gradOutput, &buffer_backward_gradOutput, lt_relu_diff_dst, lt_user_gradOutput), err );

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_relu_diff_dst), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradInput), err);

  dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_relu_diff_src);
  if(cvt_backward_gradInput_back) {
    gradInputWorkspace->cvtPrmt = cvt_backward_gradInput_back;
    gradInputWorkspace->sync = 1;
  }

  TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);

  primitives->tensor->storage->data[RELU_PRIM_CVT_GRADOUTPUT_BWD] = (long)cvt_backward_gradOutput;
  primitives->tensor->storage->data[RELU_BUFFER_GRADOUTPUT_BWD]   = (long)buffer_backward_gradOutput;
  primitives->tensor->storage->data[RELU_BUFFER_GRADINPUT_BWD]    = (long)buffer_backward_gradInput;
}

void MKLNN_(Threshold_updateOutput)(
  THMKLLongTensor *primitives,
  int initOK, 
  THMKLTensor *input,
  THMKLTensor *output,
  real threshold)
{
  struct timeval start, init, end;
  gettimeofday(&start, NULL);

  if(0 == initOK) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT] = MKL_RELU_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_RELU_BUFFER;
    MKLNN_(SpatialConvolution_Relu_init_forward)(primitives, input, output, threshold);
  }
  gettimeofday(&init, NULL);

  dnnError_t err;
  dnnPrimitive_t relu_fwd = (dnnPrimitive_t) (primitives->tensor->storage->data[RELU_PRIM_FWD]);
  real *buffer_forward_output = (real*) (primitives->tensor->storage->data[RELU_BUFFER_OUTPUT_FWD]);
  //The ouput data is modified by the computation
  real *inPtr  = TH_MKL_(data)(input);
  real *outPtr = TH_MKL_(data)(output);
  if(NULL != buffer_forward_output) {
    fprintf(stderr, "%s Fatal error\n", __func__);
    output->dnnMem = 1;
    outPtr = buffer_forward_output;
  }

  real *resRelu[dnnResourceNumber] = {0};
  resRelu[dnnResourceSrc] = inPtr;
  resRelu[dnnResourceDst] = outPtr;
  CHECK_ERR( MKLDNN_(dnnExecute)(relu_fwd, (void**)resRelu), err );

  gettimeofday(&end,NULL);
#if MKL_TIME
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	Relu MKLDNN_ time forward = %.2f ms\n",duration );
#endif

}

void MKLNN_(Threshold_updateGradInput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput)
{
  struct timeval start,end;
  gettimeofday(&start,NULL);

  if(0 == initOK) {
    MKLNN_(SpatialConvolution_Relu_init_backward)(primitives, input, gradOutput, gradInput);
  }

  dnnError_t err;
  dnnPrimitive_t relu_bwd = (dnnPrimitive_t) (primitives->tensor->storage->data[RELU_PRIM_BWD]);
  dnnPrimitive_t cvt_backward_gradOutput = (dnnPrimitive_t)primitives->tensor->storage->data[RELU_PRIM_CVT_GRADOUTPUT_BWD];
  dnnPrimitive_t cvt_backward_gradInput = (dnnPrimitive_t)primitives->tensor->storage->data[RELU_PRIM_CVT_GRADINPUT_BWD];

  real *buffer_backward_gradOutput = (real *)primitives->tensor->storage->data[RELU_BUFFER_GRADOUTPUT_BWD];
  real *buffer_backward_gradInput = (real *)primitives->tensor->storage->data[RELU_BUFFER_GRADINPUT_BWD];

  real *resRelu[dnnResourceNumber] = {0};
  real *gradInPtr = TH_MKL_(data)(gradInput);
  real *gradOutPtr = TH_MKL_(data)(gradOutput);
  if(buffer_backward_gradInput){
    fprintf(stderr, "%s Fatal error\n", __func__);
    gradInput->dnnMem = 1;
    gradInPtr = buffer_backward_gradInput;
  }

  resRelu[dnnResourceSrc]  = TH_MKL_(data)(input);
  resRelu[dnnResourceDiffSrc] = gradInPtr;
  resRelu[dnnResourceDiffDst] = gradOutPtr;

  if(cvt_backward_gradOutput) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_backward_gradOutput, gradOutPtr, buffer_backward_gradOutput), err );
    resRelu[dnnResourceDiffDst] = buffer_backward_gradOutput;
  }

  CHECK_ERR( MKLDNN_(dnnExecute)(relu_bwd, (void**)resRelu), err );

#if LOG_ENABLE | MKL_TIME
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	Relu MKLDNN_ time backward = %.2f ms\n",duration );
#endif
}

#endif

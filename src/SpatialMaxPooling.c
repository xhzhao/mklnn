#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SpatialMaxPooling.c"
#else

static void MKLNN_(SpatialMaxPooling_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *output,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode)
{
  THArgCheck(input->tensor->nDimension == 3 || input->tensor->nDimension == 4, 2, "3D or 4D (batch mode) tensor expected");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (DIMENSION == input->tensor->nDimension) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2, "input image smaller than kernel size");

  /* sizes */
  long nslices = input->size[dimh-1];
  long iheight = input->size[dimh];
  long iwidth = input->size[dimw];
  long oheight;
  long owidth;
  if (ceil_mode) {
    oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  } else {
    oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH) {
    // ensure that the last pooling starts inside the image
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  TH_MKL_(resize4d)(output, nbatch, nslices, oheight, owidth);

  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int outC = nslices;
  int outH = oheight;
  int outW = owidth;

#if LOG_ENABLE
  fprintf(stderr,"  SpatialMaxPooling_MKLDNN__init_forward start, N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d,ceilmode=%d\n",N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,ceilmode );
#endif

  dnnError_t err;
  dnnPrimitiveAttributes_t attributes = NULL;
  dnnPrimitive_t pool_fwd = NULL;
  dnnPrimitive_t pool_bwd = NULL;
  dnnLayout_t lt_user_input = NULL;
  dnnLayout_t lt_user_output = NULL;
  dnnLayout_t lt_pool_forward_output = NULL;
  dnnLayout_t lt_pool_forward_input = NULL;
  dnnLayout_t lt_pool_forward_workspace = NULL;
  dnnPrimitive_t cvt_forward_output_back = NULL;
  real *buffer_forward_output = NULL;
  real *buffer_forward_workspace = NULL;

  int input_layout_create_local = 0;
  int inputOffset[DIMENSION-2]    = {0, 0};
  size_t inputSize[DIMENSION]     = {inW, inH, inC, N};
  size_t inputStrides[DIMENSION]  = {1, inW, inH*inW, inC*inH*inW};
  size_t outputSize[DIMENSION]    = {outW, outH, outC, N};
  size_t outputStrides[DIMENSION] = {1, outW, outH*outW, outC*outH*outW};
  size_t kernelSize[2] = {kH, kW};
  size_t kernelStride[2] = {dH, dW};

  if((NULL == input->workspace) || (NULL == input->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_user_input = input->workspace->layout;
  }
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_output, DIMENSION, outputSize, outputStrides), err );

  CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesCreate)(&attributes), err );
  if(ceil_mode) {
    int pad[DIMENSION-2] = {-padW, -padH};
    CHECK_ERR( MKLDNN_(dnnPoolingCreateForward)(&pool_fwd, attributes, dnnAlgorithmPoolingMax, lt_user_input, kernelSize, kernelStride, pad, dnnBorderZeros), err );
    CHECK_ERR( MKLDNN_(dnnPoolingCreateBackward)(&pool_bwd, attributes, dnnAlgorithmPoolingMax, lt_user_input, kernelSize, kernelStride, pad, dnnBorderZeros), err );
  } else {
    int pad[DIMENSION] = {-padW, -padH, -padW, -padH};
    CHECK_ERR( MKLDNN_(dnnPoolingCreateForward)(&pool_fwd, attributes, dnnAlgorithmPoolingMax, lt_user_input, kernelSize, kernelStride, pad, dnnBorderZerosAsymm), err );
    CHECK_ERR( MKLDNN_(dnnPoolingCreateBackward)(&pool_bwd, attributes,dnnAlgorithmPoolingMax, lt_user_input, kernelSize, kernelStride, pad, dnnBorderZerosAsymm), err );
  }

  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_pool_forward_output, pool_fwd, dnnResourceDst), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_pool_forward_workspace, pool_fwd, dnnResourceWorkspace), err );
  CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)&buffer_forward_workspace, lt_pool_forward_workspace), err );

#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_pool_forward_output);
  int size2 = outW*outH*outC*N*sizeof(real);
  if(size1 == size2) {
#if 0 
    fprintf(stderr,"MKLDNN_ MaxPooling forward ouput layout match OK\n");
#endif
  } else {
    if(!MKLDNN_(dnnLayoutCompare)(lt_user_output, lt_pool_forward_output)) {
      CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_forward_output_back, lt_user_output, lt_pool_forward_output), err );
      fprintf(stderr,"MKLDNN_ MaxPooling forward ouput layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
    }
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_forward_output), lt_pool_forward_output), err );
  }
#endif

  if(attributes) {
    CHECK_ERR( MKLDNN_(dnnPrimitiveAttributesDestroy)(&attributes), err );
  }

  if(input_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_input), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_output), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_pool_forward_input), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_pool_forward_workspace), err);

  dnnWorkspace* outputWorkspace = WORKSPACE_(New)(lt_pool_forward_output);
  if(cvt_forward_output_back) {
    outputWorkspace->cvtPrmt = cvt_forward_output_back;
    outputWorkspace->sync = 1;
  }
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  //save the dnnPrimitive to THTensor(long int array)
  primitives->tensor->storage->data[POOL_PRIM_FWD]               = (long)pool_fwd;
  primitives->tensor->storage->data[POOL_PRIM_BWD]               = (long)pool_bwd;

  primitives->tensor->storage->data[POOL_BUFFER_WORKSPACE]       = (long)buffer_forward_workspace;
  primitives->tensor->storage->data[POOL_BUFFER_OUTPUT_FWD]      = (long)buffer_forward_output;

#if LOG_ENABLE
  fprintf(stderr,"  SpatialMaxPooling_MKLDNN__init_forward end.\n" );
#endif
}

static void MKLNN_(SpatialMaxPooling_init_backward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput)
{
  THArgCheck(input->tensor->nDimension == 4, 2, "4D (batch mode) tensor expected");

  TH_MKL_(resizeAs)(gradInput, input);

  int N = gradInput->size[0];
  int gradInC = gradInput->size[1];
  int gradInH = gradInput->size[2];
  int gradInW = gradInput->size[3];

  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];

#if LOG_ENABLE
  fprintf(stderr, "  SpatialMaxPooling_MKLDNN__init_backward start, N=%d, gradInC=%d, gradInH=%d, gradInW=%d, gradOutC=%d, gradOutH=%d, gradOutW=%d\n", \
                  N, gradInC, gradInH, gradInW, gradOutC, gradOutH, gradOutW);
#endif
  dnnError_t err;
  dnnLayout_t lt_user_gradInput = NULL;
  dnnLayout_t lt_user_gradOutput = NULL;
  dnnLayout_t lt_pool_backward_gradOutput = NULL;
  dnnLayout_t lt_pool_backward_gradInput = NULL;
  dnnPrimitive_t cvt_backward_gradInput_back = NULL;
  dnnPrimitive_t cvt_backward_gradOutput = NULL;
  real *buffer_backward_gradInput = NULL;
  real *buffer_backward_gradOutput = NULL;

  int gradOutput_layout_create_local = 0;
  int gradInputOffset[DIMENSION - 2 ] = {0, 0};
  size_t gradInputSize[DIMENSION]     = {gradInW, gradInH, gradInC, N};
  size_t gradInputStrides[DIMENSION]  = {1, gradInW, gradInH*gradInW, gradInC*gradInH*gradInW};
  size_t gradOutputSize[DIMENSION]    = {gradOutW, gradOutH, gradOutC, N};
  size_t gradOutputStrides[DIMENSION] = {1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW};

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }
  CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradInput, DIMENSION, gradInputSize, gradInputStrides), err );

  dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->tensor->storage->data[POOL_PRIM_BWD]);
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_pool_backward_gradInput, pool_bwd, dnnResourceDiffSrc), err );
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_pool_backward_gradOutput, pool_bwd, dnnResourceDiffDst), err );

  CHECK_ERR( MKLNN_(init_conversion)(&cvt_backward_gradOutput, &buffer_backward_gradOutput, lt_pool_backward_gradOutput, lt_user_gradOutput), err );

#if MKL_BUFFER_DBG
  int size1 = MKLDNN_(dnnLayoutGetMemorySize)(lt_pool_backward_gradInput);
  int size2 = gradInW*gradInH*gradInC*N*sizeof(real);
  if(size1 == size2) {
#if 0
    fprintf(stderr,"MKLDNN_ MaxPooling bwddata input layout match OK\n");
#endif
  } else {
    if(!MKLDNN_(dnnLayoutCompare)(lt_user_gradInput, lt_pool_backward_gradInput)) {
      CHECK_ERR( MKLDNN_(dnnConversionCreate)(&cvt_backward_gradInput_back, lt_user_gradInput, lt_pool_backward_gradInput), err );
      fprintf(stderr,"MKLDNN_ MaxPooling bwddata input layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
    }
    CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_backward_gradInput), lt_pool_backward_gradInput), err );
  }
#endif

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradInput), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_pool_backward_gradOutput), err);

  dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_pool_backward_gradInput);
  if(cvt_backward_gradInput_back) {
    gradInputWorkspace->cvtPrmt = cvt_backward_gradInput_back;
    gradInputWorkspace->sync = 1;
  }
  TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);
  //save the dnnPrimitive to THTensor(long int array)

  primitives->tensor->storage->data[POOL_PRIM_CVT_GRADOUTPUT_BWD]    = (long)cvt_backward_gradOutput;

  primitives->tensor->storage->data[POOL_BUFFER_GRADINPUT_BWD]       = (long)buffer_backward_gradInput;
  primitives->tensor->storage->data[POOL_BUFFER_GRADOUTPUT_BWD]      = (long)buffer_backward_gradOutput;

#if LOG_ENABLE
  fprintf(stderr,"  SpatialMaxPooling_MKLDNN__init_backward end.\n" );
#endif
}

void MKLNN_(SpatialMaxPooling_updateOutput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *output,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode)
{
  struct timeval start, init, end;
  gettimeofday(&start, NULL);
  if(0 == initOK) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT]   = MKL_POOL_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_POOL_BUFFER;
    MKLNN_(SpatialMaxPooling_init_forward)(primitives, input, output, kH, kW, dH, dW, padH, padW, ceil_mode);
  }
  gettimeofday(&init, NULL);

  dnnError_t err;
  dnnPrimitive_t pool_fwd = (dnnPrimitive_t) (primitives->tensor->storage->data[POOL_PRIM_FWD]);
  real *buffer_workspace = (real*) (primitives->tensor->storage->data[POOL_BUFFER_WORKSPACE]);
  real *buffer_forward_output = (real*) (primitives->tensor->storage->data[POOL_BUFFER_OUTPUT_FWD]);
  //The ouput data is modified by the computation
  real *inPtr     = TH_MKL_(data)(input);
  real *outPtr    = TH_MKL_(data)(output);
  if(NULL != buffer_forward_output) {
    fprintf(stderr, "%s Fatal error\n", __func__);
    output->dnnMem = 1;
    outPtr = buffer_forward_output;
  }

  real *resPool[dnnResourceNumber] = {0};
  resPool[dnnResourceSrc] = inPtr;
  resPool[dnnResourceDst] = outPtr;
  resPool[dnnResourceWorkspace] = buffer_workspace;

  CHECK_ERR( MKLDNN_(dnnExecute)(pool_fwd, (void*)resPool), err );

  gettimeofday(&end,NULL);
#if LOG_ENABLE || MKL_TIME
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"  MaxPooling MKLDNN_ time forward = %.2f ms\n",duration );
#endif
}

void MKLNN_(SpatialMaxPooling_updateGradInput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput)
{
  struct timeval start,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  if(initOK == 0) {
    MKLNN_(SpatialMaxPooling_init_backward)(primitives, input, gradOutput, gradInput);
  }
  dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->tensor->storage->data[POOL_PRIM_BWD]);
  dnnPrimitive_t cvt_backward_gradOutput = (dnnPrimitive_t) (primitives->tensor->storage->data[POOL_PRIM_CVT_GRADOUTPUT_BWD]);
  real * buffer_backward_gradInput = (real *) (primitives->tensor->storage->data[POOL_BUFFER_GRADINPUT_BWD]);
  real * buffer_backward_gradOutput = (real *) (primitives->tensor->storage->data[POOL_BUFFER_GRADOUTPUT_BWD]);
  real * buffer_workspace = (real *) (primitives->tensor->storage->data[POOL_BUFFER_WORKSPACE]);

  THTensor_(zero)(gradInput->tensor);
  real *gradInput_data = TH_MKL_(data)(gradInput);
  real *gradOutput_data = TH_MKL_(data)(gradOutput);

  if(buffer_backward_gradInput){
    fprintf(stderr, "%s Fatal error\n", __func__);
    gradInput->dnnMem = 1;
    gradInput_data = buffer_backward_gradInput;
  }

  real *resPool[dnnResourceNumber] = {0};
  resPool[dnnResourceDiffSrc] = gradInput_data;
  resPool[dnnResourceWorkspace] = buffer_workspace;

  if(cvt_backward_gradOutput) {
    resPool[dnnResourceDiffDst] = buffer_backward_gradOutput;
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_backward_gradOutput, gradOutput_data, resPool[dnnResourceDiffDst]), err );
  } else {
    resPool[dnnResourceDiffDst] = gradOutput_data;
  }

  CHECK_ERR( MKLDNN_(dnnExecute)(pool_bwd, (void*)resPool), err );

#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"  MaxPooling MKLDNN_ time backward = %.2f ms\n",duration );
#endif
 

}

#endif

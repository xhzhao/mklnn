#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SpatialCrossMapLRN.c"
#else

static void MKLNN_(CrossChannelLRN_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input, 
  THMKLTensor *output,
  int size,
  float alpha,
  float beta,
  float k)
{
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];
  size_t inputSize[DIMENSION] =	{inW, inH, inC, N};
  size_t inputStrides[DIMENSION] = {1, inW, inH*inW, inC*inH*inW};

  TH_MKL_(resizeAs)(output,input);

#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__init_forward start, N=%d,C=%d,H=%d,W=%d, size = %d, alpha = %.2f, beta = %.2f, k = %.2f \n", N,inC,inH,inW,size,alpha,beta,k);
#endif

  dnnError_t err;
  dnnPrimitive_t lrn_forward = NULL;
  dnnPrimitive_t lrn_backward = NULL;
  dnnLayout_t lt_user_input = NULL;
  dnnLayout_t lt_lrn_workspace = NULL;
  dnnLayout_t lt_lrn_forward_output = NULL;
  real *buffer_workspace = NULL;
  int input_layout_create_local = 0;

  if((0 == input->workspace) || (NULL == input->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_user_input = input->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnLRNCreateForward)(&lrn_forward, NULL, lt_user_input, size, alpha, beta, k), err );
  CHECK_ERR( MKLDNN_(dnnLRNCreateBackward)(&lrn_backward, NULL, lt_user_input, lt_user_input, size, alpha, beta, k), err );

  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_lrn_workspace, lrn_forward, dnnResourceWorkspace);
  CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_workspace), lt_lrn_workspace), err );

  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_lrn_forward_output, lrn_forward, dnnResourceDst);

  if(input_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_input), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_lrn_workspace), err);

  dnnWorkspace* outputWorkspace = WORKSPACE_(New)(lt_lrn_forward_output);
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  primitives->tensor->storage->data[LRN_PRIM_FWD] = (long)lrn_forward;
  primitives->tensor->storage->data[LRN_PRIM_BWD] = (long)lrn_backward;
  primitives->tensor->storage->data[LRN_BUFFER_WORKSPACE] = (long)buffer_workspace;
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__init_forward end.\n");
#endif
}

static void MKLNN_(CrossChannelLRN_init_backward)(
  THMKLLongTensor *primitives,
  THMKLTensor *gradOutput, 
  THMKLTensor *gradInput)
{
  int N = gradOutput->size[0];
  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];

  dnnError_t err;
  dnnPrimitive_t lrn_backward = (dnnPrimitive_t)primitives->tensor->storage->data[LRN_PRIM_BWD];
  dnnLayout_t lt_user_gradOutput;
  dnnLayout_t lt_lrn_backward_gradOutput=NULL;
  dnnPrimitive_t cvt_backward_gradOutput = NULL;
  dnnLayout_t lt_lrn_backward_gradInput = NULL;
  real *buffer_backward_gradOutput = NULL;

  int gradOutput_layout_create_local = 0;
  size_t gradOutputSize[DIMENSION] = {gradOutW, gradOutH, gradOutC, N};
  size_t gradOutputStrides[DIMENSION] = {1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW};

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }

  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_lrn_backward_gradOutput, lrn_backward, dnnResourceDiffDst);
  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_lrn_backward_gradInput, lrn_backward, dnnResourceDiffSrc);
  //backward conversion init
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_backward_gradOutput, &buffer_backward_gradOutput, lt_lrn_backward_gradOutput, lt_user_gradOutput), err );

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_lrn_backward_gradOutput), err);

  dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_lrn_backward_gradInput);
  TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);

  //save the dnnPrimitive to THTensor(long int array)
  primitives->tensor->storage->data[LRN_PRIM_CVT_GRADOUTPUT_BWD] = (long)cvt_backward_gradOutput;
  primitives->tensor->storage->data[LRN_BUFFER_GRADOUTPUT_BWD] = (long)buffer_backward_gradOutput;
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__init_backward end.\n");
#endif
}

void MKLNN_(CrossChannelLRN_updateOutput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input, 
  THMKLTensor *output,
  int size, 
  float alpha, 
  float beta, 
  float k)

{
#if LOG_ENABLE
  fprintf(stderr, "BatchNormalization_MKLDNN__updateOutput start.\n");
#endif
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);

  if(initOK == 0) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT] = MKL_LRN_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_LRN_BUFFER;
    MKLNN_(CrossChannelLRN_init_forward)(primitives, input, output, size, alpha, beta, k);
  }

  dnnError_t err;
  dnnPrimitive_t lrn_forward = (dnnPrimitive_t)primitives->tensor->storage->data[LRN_PRIM_FWD];
  real *buffer_workspace = (real *)primitives->tensor->storage->data[LRN_BUFFER_WORKSPACE];

  void* LRN_res[dnnResourceNumber];
  LRN_res[dnnResourceSrc] = TH_MKL_(data)(input);
  LRN_res[dnnResourceDst] = TH_MKL_(data)(output);
  LRN_res[dnnResourceWorkspace] = buffer_workspace;

  /*LRN is usually following conv(+relu) and won't change the layout*/
  CHECK_ERR( MKLDNN_(dnnExecute)(lrn_forward, (void*)LRN_res), err );

#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	LRN MKLDNN_ forward time = %.2f ms \n",duration1);
#endif

#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__updateOutput end.\n");
#endif
}


void MKLNN_(CrossChannelLRN_backward)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input, 
  THMKLTensor *gradOutput, 
  THMKLTensor *gradInput)
{
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__backward start.\n");
#endif
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);
  TH_MKL_(resizeAs)(gradInput, input);
  if(0 == initOK) {
    MKLNN_(CrossChannelLRN_init_backward)(primitives, gradOutput, gradInput);
  }

  dnnError_t err;
  dnnPrimitive_t lrn_backward = (dnnPrimitive_t)primitives->tensor->storage->data[LRN_PRIM_BWD];
  dnnPrimitive_t cvt_backward_gradOutput = (dnnPrimitive_t) (primitives->tensor->storage->data[LRN_PRIM_CVT_GRADOUTPUT_BWD]);
  real *buffer_backward_gradOutput = (real *)(primitives->tensor->storage->data[LRN_BUFFER_GRADOUTPUT_BWD]);
  real *buffer_workspace = (real *)primitives->tensor->storage->data[LRN_BUFFER_WORKSPACE];

  void* LRN_res[dnnResourceNumber] = {0};
  LRN_res[dnnResourceSrc] = TH_MKL_(data)(input);
  LRN_res[dnnResourceDiffDst] = TH_MKL_(data)(gradOutput);
  LRN_res[dnnResourceDiffSrc] = TH_MKL_(data)(gradInput);
  LRN_res[dnnResourceWorkspace] = buffer_workspace;

  if(cvt_backward_gradOutput) {
    CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_backward_gradOutput, TH_MKL_(data)(gradOutput), buffer_backward_gradOutput), err );
    LRN_res[dnnResourceDiffDst] = buffer_backward_gradOutput;
  }

  CHECK_ERR( MKLDNN_(dnnExecute)(lrn_backward, LRN_res), err );

#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	LRN MKLDNN_ backward time = %.2f ms \n",duration1);
#endif

#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN__backward end.\n");
#endif
}

#endif

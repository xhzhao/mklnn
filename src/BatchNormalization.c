#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/BatchNormalization.c"
#else

//#include "MKLDNN_.h"

static void MKLNN_(BatchNormalization_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor *input,
  THMKLTensor *output,
  double eps)
{
  TH_MKL_(resizeAs)(output,input);
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];
  size_t inputSize[DIMENSION] = {inW, inH, inC, N};
  size_t inputStrides[DIMENSION] = { 1, inW, inH*inW, inC*inH*inW };

  dnnError_t err;
  dnnPrimitive_t bn_forward = NULL;
  dnnPrimitive_t bn_backward_data = NULL;
  dnnPrimitive_t bn_bwd_scaleshift = NULL;

  dnnLayout_t lt_user_input = NULL;
  real *buffer_forward_workspace = NULL;
  real *buffer_forward_scaleshift = NULL;
  dnnLayout_t lt_bn_forward_workspace = NULL;
  dnnLayout_t lt_bn_forward_scaleshift = NULL;
  dnnLayout_t lt_bn_forward_output = NULL;
  int input_layout_create_local = 0;

  if(0 == input->workspace || (0 == input->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides), err );
    input_layout_create_local = 1;
  } else {
    lt_user_input = input->workspace->layout;
  }

  CHECK_ERR( MKLDNN_(dnnBatchNormalizationCreateForward)(&bn_forward, NULL, lt_user_input, eps), err );
  CHECK_ERR( MKLDNN_(dnnBatchNormalizationCreateBackwardData)(&bn_backward_data, NULL, lt_user_input, eps), err );
  CHECK_ERR( MKLDNN_(dnnBatchNormalizationCreateBackwardScaleShift)(&bn_bwd_scaleshift, NULL, lt_user_input, eps), err );

  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bn_forward_workspace, bn_forward, dnnResourceWorkspace);
  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bn_forward_output, bn_forward, dnnResourceDst);
  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bn_forward_scaleshift, bn_forward, dnnResourceScaleShift);

  CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_forward_workspace), lt_bn_forward_workspace), err );
  CHECK_ERR( MKLDNN_(dnnAllocateBuffer)((void**)(&buffer_forward_scaleshift), lt_bn_forward_scaleshift), err );

  if(input_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_input), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bn_forward_scaleshift), err);
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bn_forward_workspace), err);

  dnnWorkspace* outputWorkspace = WORKSPACE_(New)(lt_bn_forward_output);
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  //save the dnnPrimitive to THTensor(long int array)
  primitives->tensor->storage->data[BN_PRIM_FWD] = (long)bn_forward;
  primitives->tensor->storage->data[BN_PRIM_BWD_DATA] = ( long)bn_backward_data;
  primitives->tensor->storage->data[BN_PRIM_BWD_SCALESHIFT] = (long)bn_bwd_scaleshift;

  primitives->tensor->storage->data[BN_BUFFER_WORKSPACE] = (long)buffer_forward_workspace;
  primitives->tensor->storage->data[BN_BUFFER_SCALESHIFT] = (long)buffer_forward_scaleshift;
}

static void MKLNN_(BatchNormalization_init_backward)(
  THMKLLongTensor *primitives,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput
  )
{
  int N = gradOutput->size[0];
  int gradOutC = gradOutput->size[1];
  int gradOutH = gradOutput->size[2];
  int gradOutW = gradOutput->size[3];
  size_t gradOutputSize[DIMENSION] = {gradOutW, gradOutH, gradOutC, N};
  size_t gradOutputStrides[DIMENSION] = { 1, gradOutW, gradOutH*gradOutW, gradOutC*gradOutH*gradOutW };

  dnnError_t err;
  dnnPrimitive_t bn_backward_data = (dnnPrimitive_t)primitives->tensor->storage->data[BN_PRIM_BWD_DATA];
  dnnLayout_t lt_user_gradOutput = NULL;
  dnnLayout_t lt_bn_backward_data_gradOutput = NULL;
  dnnPrimitive_t cvt_backward_gradOutput = NULL;
  real * buffer_backward_gradOutput = NULL;
  dnnLayout_t lt_bn_backward_data_gradInput = NULL;
  int gradOutput_layout_create_local = 0;

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides), err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }
  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bn_backward_data_gradOutput, bn_backward_data, dnnResourceDiffDst);
  MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_bn_backward_data_gradInput, bn_backward_data, dnnResourceDiffSrc);

  //backward conversion init
  CHECK_ERR( MKLNN_(init_conversion)(&cvt_backward_gradOutput, &buffer_backward_gradOutput, lt_bn_backward_data_gradOutput, lt_user_gradOutput), err );

  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput), err);
  }
  CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_bn_backward_data_gradOutput), err);
  dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_bn_backward_data_gradInput);
  TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);

  //save the dnnPrimitive to THTensor(long int array)
  primitives->tensor->storage->data[BN_PRIM_CVT_GRADOUTPUT_BWD] = (long)cvt_backward_gradOutput;
  primitives->tensor->storage->data[BN_BUFFER_GRADOUTPUT_BWD] = (long)buffer_backward_gradOutput;
}

void MKLNN_(BatchNormalization_updateOutput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *output,
  THTensor *weight,
  THTensor *bias,
  THTensor *running_mean,
  THTensor *running_var,
  bool train,
  double momentum,
  double eps
  )
{ 
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);

  if(initOK == 0) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT] = MKL_BN_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_BN_BUFFER;
    MKLNN_(BatchNormalization_init_forward)(primitives, input, output, eps);
  }
  gettimeofday(&mid,NULL);

  dnnError_t err;
  dnnPrimitive_t bn_forward = (dnnPrimitive_t)primitives->tensor->storage->data[BN_PRIM_FWD];
  real *buffer_forward_workspace = (real *)primitives->tensor->storage->data[BN_BUFFER_WORKSPACE];
  real *buffer_forward_scaleshift = (real *)primitives->tensor->storage->data[BN_BUFFER_SCALESHIFT];

  int i = 0;
  long nInput = input->size[1];
  for(; i < nInput; i++) {
    buffer_forward_scaleshift[i] = weight ? THTensor_(get1d)(weight, i) : 1;
    buffer_forward_scaleshift[i+nInput] = bias ? THTensor_(get1d)(bias, i) : 0;
  }

  void* BatchNorm_res[dnnResourceNumber] = {0};
  BatchNorm_res[dnnResourceSrc] = TH_MKL_(data)(input);
  BatchNorm_res[dnnResourceDst] = TH_MKL_(data)(output);
  BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
  BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

  CHECK_ERR( MKLDNN_(dnnExecute)(bn_forward, (void*)BatchNorm_res), err );

#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
  fprintf(stderr,"	BatchNorm MKLDNN_ forward time1 = %.2f ms, time2 = %.2f ms \n",duration1,duration2);
#endif
}

void MKLNN_(BatchNormalization_backward)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *weight
  )
{
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);

  long nInput = THTensor_(size)(input->tensor, 1);
  TH_MKL_(resizeAs)(gradInput, input);

  dnnError_t err;
  int gradInC = gradInput->size[1];
  dnnPrimitive_t bn_backward_data   = (dnnPrimitive_t)primitives->tensor->storage->data[BN_PRIM_BWD_DATA];
  dnnPrimitive_t bn_bwd_scaleshift 	= (dnnPrimitive_t)primitives->tensor->storage->data[BN_PRIM_BWD_SCALESHIFT];
  real *buffer_forward_workspace 	= (real *)primitives->tensor->storage->data[BN_BUFFER_WORKSPACE];
  real *buffer_forward_scaleshift 	= (real *)primitives->tensor->storage->data[BN_BUFFER_SCALESHIFT];
  real *buffer_backward_gradOutput = (real *)(primitives->tensor->storage->data[BN_BUFFER_GRADOUTPUT_BWD]);

  real* gradOutPtr = TH_MKL_(data)(gradOutput);
  real* gradInPtr = TH_MKL_(data)(gradInput);
  void* BatchNorm_res[dnnResourceNumber] = {0};
  BatchNorm_res[dnnResourceSrc] = TH_MKL_(data)(input);
  BatchNorm_res[dnnResourceDiffDst] = gradOutPtr;
  BatchNorm_res[dnnResourceDiffSrc] = gradInPtr;
  BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
  BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;

  if(gradInput == 0) {
    CHECK_ERR( MKLDNN_(dnnExecute)(bn_bwd_scaleshift, (void*)BatchNorm_res), err );
    int i = 0;
    for(; i < gradInC; i++) {
      THTensor_(set1d)(gradWeight, i, buffer_forward_scaleshift[i]);
      THTensor_(set1d)(gradBias, i, buffer_forward_scaleshift[i+gradInC]);
    }
  } else {
    if(initOK == 0) {
      MKLNN_(BatchNormalization_init_backward)(primitives, gradOutput, gradInput);
    }
    dnnPrimitive_t cvt_backward_gradOutput = (dnnPrimitive_t) (primitives->tensor->storage->data[BN_PRIM_CVT_GRADOUTPUT_BWD]);
    if(cvt_backward_gradOutput) {
      CHECK_ERR( MKLDNN_(dnnConversionExecute)(cvt_backward_gradOutput, gradOutPtr, buffer_backward_gradOutput), err );
      BatchNorm_res[dnnResourceDiffDst] = buffer_backward_gradOutput;
    }
    gettimeofday(&mid,NULL);
    CHECK_ERR( MKLDNN_(dnnExecute)(bn_backward_data, (void*)BatchNorm_res), err );
  }
#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
  fprintf(stderr,"        BatchNorm MKLDNN_ backward time1 = %.2f ms, time2 = %.2f ms \n",duration1,duration2);
#endif
}

#endif

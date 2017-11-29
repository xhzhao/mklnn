#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/Concat.c"
#else

static void MKLNN_(Concat_init_forward)(
  THMKLLongTensor *primitives,
  THMKLTensor **inputArray,
  THMKLTensor *output,
  int moduleNum
  )
{
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__init_forward start\n");
#endif
  dnnError_t err;
  dnnPrimitive_t m_concat_forward = NULL;
  THMKLTensor *input = NULL;
  dnnLayout_t *layouts = malloc(moduleNum*sizeof(dnnLayout_t));
  int* create_layout_local = malloc(moduleNum*sizeof(int));
  int i;
  for(i = 0; i < moduleNum; ++i) {
    input = inputArray[i];
    create_layout_local[i] = 0;
    if((0 == input->workspace) || (0 == input->workspace->layout)) {
      int N = input->size[0];
      int inC = input->size[1];
      int inH = input->size[2];
      int inW = input->size[3];
      dnnLayout_t lt_user_input = NULL;
      size_t inputSize[DIMENSION]    = {inW, inH, inC, N};
      size_t inputStrides[DIMENSION] = {1, inW, inH*inW, inC*inH*inW};
      CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_input, DIMENSION, inputSize, inputStrides) , err );
      layouts[i] = lt_user_input;
      create_layout_local[i] = 1;
    } else {
      layouts[i] = input->workspace->layout;
    }
  }
  CHECK_ERR( MKLDNN_(dnnConcatCreate)(&m_concat_forward, NULL, moduleNum, layouts), err);

  dnnLayout_t lt_concat_forward_output = NULL;
  CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_concat_forward_output, m_concat_forward, dnnResourceDst), err );

  for(i = 0; i< moduleNum; ++i) {
    if(create_layout_local[i]) {
      CHECK_ERR( MKLDNN_(dnnLayoutDelete)(layouts[i]) , err );
    }
  }
  free(create_layout_local);
  free(layouts);
  dnnWorkspace* outputWorkspace = WORKSPACE_(New)(lt_concat_forward_output);
  TH_MKL_(changeWorkspace)(output, outputWorkspace);

  primitives->tensor->storage->data[CONCAT_PRIM_FWD] = (long)m_concat_forward;

#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__init_forward end. \n");
#endif
}

static void MKLNN_(Concat_init_backward)(
          THMKLLongTensor *primitives,
          THMKLTensor **gradInputArray,
          THMKLTensor *gradOutput,
          int moduleNum)

{
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__init_backward start. gradarray = 0x%x, gradOutput = 0x%d, moduleNum = %d,  primitives = 0x%x\n", gradarray, gradOutput, moduleNum, primitives);
#endif
  struct timeval start,end;
  gettimeofday(&start,NULL);
  dnnError_t err;

  dnnPrimitive_t concat_split = NULL;
  dnnLayout_t lt_user_gradOutput = NULL;
  int gradOutput_layout_create_local = 0;

  if((0 == gradOutput->workspace) || (0 == gradOutput->workspace->layout)) {
    //create NCHW layout here
    int N = gradOutput->size[0];
    int gradOutputC = gradOutput->size[1];
    int gradOutputH = gradOutput->size[2];
    int gradOutputW = gradOutput->size[3];
    size_t gradOutputSize[DIMENSION]    = {gradOutputW, gradOutputH, gradOutputC, N};
    size_t gradOutputStrides[DIMENSION] = {1, gradOutputW, gradOutputH *gradOutputW, gradOutputC*gradOutputH*gradOutputW};
    CHECK_ERR( MKLDNN_(dnnLayoutCreate)(&lt_user_gradOutput, DIMENSION, gradOutputSize, gradOutputStrides) , err );
    gradOutput_layout_create_local = 1;
  } else {
    lt_user_gradOutput = gradOutput->workspace->layout;
  }

  THMKLTensor *gradInput = NULL;
  size_t* split_channels = malloc(moduleNum*sizeof(size_t*)); 
  int i;
  for(i = 0; i < moduleNum; ++i){
    gradInput = gradInputArray[i];
    split_channels[i] = gradInput->size[1];
  }

  CHECK_ERR( MKLDNN_(dnnSplitCreate)(&concat_split, NULL, moduleNum, lt_user_gradOutput, split_channels), err);

  dnnLayout_t lt_concat_gradInput = NULL;
  for(i = 0; i < moduleNum; ++i){
    CHECK_ERR( MKLDNN_(dnnLayoutCreateFromPrimitive)(&lt_concat_gradInput, concat_split, dnnResourceMultipleDst+i), err );
    dnnWorkspace* gradInputWorkspace = WORKSPACE_(New)(lt_concat_gradInput);
    gradInput = gradInputArray[i];
    TH_MKL_(changeWorkspace)(gradInput, gradInputWorkspace);
  }
  if(gradOutput_layout_create_local) {
    CHECK_ERR( MKLDNN_(dnnLayoutDelete)(lt_user_gradOutput) , err );
  }
  free(split_channels);

  primitives->tensor->storage->data[CONCAT_PRIM_BWD] = (long)concat_split;
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__init_backward end. \n");
#endif

}

void MKLNN_(Concat_updateOutput)(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLTensor **inputArray,
  THMKLTensor *output,
  int moduleNum
  )
{
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__updateOutput start. inputarray = 0x%x, output = 0x%d, moduleNum = %d,  primitives = 0x%x, initOK = %d \n", inputarray, output, moduleNum, primitives, initOK);
#endif

  struct timeval start,end;
  gettimeofday(&start, NULL);

  if(0 == initOK) {
    primitives->tensor->storage->data[MKL_INFO_TYPE] = FLOAT_TYPE;
    primitives->tensor->storage->data[MKL_INFO_PRMT]   = MKL_CONCAT_PRMT;
    primitives->tensor->storage->data[MKL_INFO_BUFFER] = MKL_CONCAT_BUFFER;
    MKLNN_(Concat_init_forward)(primitives, inputArray, output, moduleNum);
  }

  dnnError_t err;
  dnnPrimitive_t m_concat_forward = (dnnPrimitive_t) (primitives->tensor->storage->data[CONCAT_PRIM_FWD]);
  void *concat_res[dnnResourceNumber] = {0};
  THMKLTensor *input = NULL;
  dnnLayout_t *layouts = NULL;
  int i;
  for(i=0; i < moduleNum; i++) {
    input = (THMKLTensor *)inputArray[i];
    concat_res[dnnResourceMultipleSrc+i] = TH_MKL_(data)(input);
  }
  concat_res[dnnResourceDst] = TH_MKL_(data)(output);
  CHECK_ERR( MKLDNN_(dnnExecute)(m_concat_forward, (void*)concat_res), err );
   
#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"  Concat MKLDNN_ time forward = %.2f ms\n",duration );
#endif

#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__updateOutput end. \n");
#endif

}

void MKLNN_(Concat_backward_split)(
          THMKLLongTensor *primitives,
          int initOK,
          THMKLTensor ** gradInputArray,
          THMKLTensor *gradOutput,
          int moduleNum)
{
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__backward_split start. gradarray = 0x%x, gradOutput = 0x%d, moduleNum = %d,  primitives = 0x%x, initOK = %d \n", gradarray, gradOutput, moduleNum, primitives, initOK);
#endif
  struct timeval start,end;
  gettimeofday(&start,NULL);

  if(0 == initOK) {
    MKLNN_(Concat_init_backward)(primitives, gradInputArray, gradOutput, moduleNum);
  }

  dnnError_t err;
  dnnLayout_t layout = NULL;
  dnnPrimitive_t concat_split = (dnnPrimitive_t)primitives->tensor->storage->data[CONCAT_PRIM_BWD];
  
  THMKLTensor *gradInput = NULL;
  void *split_res[dnnResourceNumber] = {0};
  split_res[dnnResourceSrc] = TH_MKL_(data)(gradOutput);
  int i;
  for(i=0; i < moduleNum; ++i) {
    gradInput = gradInputArray[i];
    split_res[dnnResourceMultipleDst+i] = TH_MKL_(data)(gradInput);
  }

  CHECK_ERR(MKLDNN_(dnnExecute)(concat_split, split_res), err);

#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"  Concat MKLDNN_ time backward = %.2f ms\n",duration );
#endif
#if LOG_ENABLE
  fprintf(stderr, "Concat_MKLDNN__backward_split end. \n");
#endif

}

#endif

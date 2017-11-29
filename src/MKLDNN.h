#ifndef _TORCH_MKLDNN_H
#define _TORCH_MKLDNN_H

#define LOG_ENABLE 	     0
#define CONVERSION_LOG	 0
#define MKL_TIME         0
#define NEW_INTERFACE    1
#define DIMENSION        4
#define MKL_BUFFER_DBG   1

/*
 * U2F:  [user] layout(which is passed in from other layers) to a layout which is to be used in [forward] operation
 * U2BD: [user] layout(which is passed in from other layers) to a layout which is to be used in [backward data] operation
 * U2BF: [user] layout(which is passed in from other layers) to a layout which is to be used in [backward filter] operation
 * U2BB: [user] layout(which is passed in from other layers) to a layout which is to be used in [backward bias] operation
 * F2BD: a layout which is to be used in [forward] operation to a layout which is to be used in [backward data] operation
 * as so on
 */

#define MKL_INFO_TYPE     0
#define MKL_INFO_PRMT     1
#define MKL_INFO_BUFFER   2
#define MKL_INFO_MAINTAIN 3

#define MKL_CONV_PRMT     18
#define MKL_CONV_BUFFER   12

#define MKL_POOL_PRMT     6
#define MKL_POOL_BUFFER   5

#define MKL_LRN_PRMT      3
#define MKL_LRN_BUFFER    2

#define MKL_BN_PRMT       4
#define MKL_BN_BUFFER     3

#define MKL_RELU_PRMT     4
#define MKL_RELU_BUFFER   3

#define MKL_CONCAT_PRMT   2
#define MKL_CONCAT_BUFFER 0

typedef enum {
   CONV_PRIM_FWD                          = MKL_INFO_MAINTAIN+0,
   CONV_PRIM_BWD_DATA                     = MKL_INFO_MAINTAIN+1,
   CONV_PRIM_BWD_FILTER                   = MKL_INFO_MAINTAIN+2,
   CONV_PRIM_BWD_BIAS                     = MKL_INFO_MAINTAIN+3,

   CONV_PRIM_CVT_INPUT_U2F                = MKL_INFO_MAINTAIN+4,
   CONV_PRIM_CVT_FILTER_U2F               = MKL_INFO_MAINTAIN+5,
   CONV_PRIM_CVT_BIAS_U2F                 = MKL_INFO_MAINTAIN+6,
   CONV_PRIM_CVT_OUTPUT_U2F               = MKL_INFO_MAINTAIN+7,

   CONV_PRIM_CVT_GRADINPUT_U2BD           = MKL_INFO_MAINTAIN+8,
   CONV_PRIM_CVT_FILTER_U2BD              = MKL_INFO_MAINTAIN+9,
   CONV_PRIM_CVT_GRADOUTPUT_U2BD          = MKL_INFO_MAINTAIN+10,

   CONV_PRIM_CVT_INPUT_U2BF               = MKL_INFO_MAINTAIN+11,
   CONV_PRIM_CVT_GRADFILTER_U2BF          = MKL_INFO_MAINTAIN+12,
   CONV_PRIM_CVT_GRADOUTPUT_U2BF          = MKL_INFO_MAINTAIN+13,
   CONV_PRIM_CVT_GRADFILTER_BF2U          = MKL_INFO_MAINTAIN+14,

   CONV_PRIM_CVT_GRADBIAS_U2BB            = MKL_INFO_MAINTAIN+15,
   CONV_PRIM_CVT_GRADOUTPUT_U2BB          = MKL_INFO_MAINTAIN+16,
   CONV_PRIM_CVT_GRADBIAS_BB2U            = MKL_INFO_MAINTAIN+17,

   CONV_BUFFER_INPUT_FWD                  = MKL_INFO_MAINTAIN+18,
   CONV_BUFFER_FILTER_FWD                 = MKL_INFO_MAINTAIN+19,
   CONV_BUFFER_BIAS_FWD                   = MKL_INFO_MAINTAIN+20,
   CONV_BUFFER_OUTPUT_FWD                 = MKL_INFO_MAINTAIN+21,

   CONV_BUFFER_GRADINPUT_BWD_DATA         = MKL_INFO_MAINTAIN+22,
   CONV_BUFFER_FILTER_BWD_DATA            = MKL_INFO_MAINTAIN+23,
   CONV_BUFFER_GRADOUTPUT_BWD_DATA        = MKL_INFO_MAINTAIN+24,

   CONV_BUFFER_INPUT_BWD_FILTER           = MKL_INFO_MAINTAIN+25,
   CONV_BUFFER_GRADFILTER_BWD_FILTER      = MKL_INFO_MAINTAIN+26,
   CONV_BUFFER_GRADOUTPUT_BWD_FILTER      = MKL_INFO_MAINTAIN+27,

   CONV_BUFFER_GRADBIAS_BWD_BIAS          = MKL_INFO_MAINTAIN+28,
   CONV_BUFFER_GRADOUTPUT_BWD_BIAS        = MKL_INFO_MAINTAIN+29

} mkldnnConvolutionIndex_t;

typedef enum {
   POOL_PRIM_FWD                          = MKL_INFO_MAINTAIN+0,
   POOL_PRIM_BWD                          = MKL_INFO_MAINTAIN+1,

   POOL_PRIM_CVT_INPUT_FWD                = MKL_INFO_MAINTAIN+2,
   POOL_PRIM_CVT_OUTPUT_FWD               = MKL_INFO_MAINTAIN+3,
   POOL_PRIM_CVT_GRADINPUT_BWD            = MKL_INFO_MAINTAIN+4,
   POOL_PRIM_CVT_GRADOUTPUT_BWD           = MKL_INFO_MAINTAIN+5,

   POOL_BUFFER_WORKSPACE                  = MKL_INFO_MAINTAIN+6,
   POOL_BUFFER_INPUT_FWD                  = MKL_INFO_MAINTAIN+7,
   POOL_BUFFER_OUTPUT_FWD                 = MKL_INFO_MAINTAIN+8,
   POOL_BUFFER_GRADINPUT_BWD              = MKL_INFO_MAINTAIN+9,
   POOL_BUFFER_GRADOUTPUT_BWD             = MKL_INFO_MAINTAIN+10
} mkldnnPoolingIndex_t;

typedef enum {
   RELU_PRIM_FWD                          = MKL_INFO_MAINTAIN+0,
   RELU_PRIM_BWD                          = MKL_INFO_MAINTAIN+1,
   RELU_PRIM_CVT_GRADOUTPUT_BWD           = MKL_INFO_MAINTAIN+2,
   RELU_PRIM_CVT_GRADINPUT_BWD            = MKL_INFO_MAINTAIN+3,

   RELU_BUFFER_OUTPUT_FWD                 = MKL_INFO_MAINTAIN+4,
   RELU_BUFFER_GRADINPUT_BWD              = MKL_INFO_MAINTAIN+5,
   RELU_BUFFER_GRADOUTPUT_BWD             = MKL_INFO_MAINTAIN+6
} mkldnnReLUIndex_t;

typedef enum {
   BN_PRIM_FWD                            = MKL_INFO_MAINTAIN+0,
   BN_PRIM_BWD_DATA                       = MKL_INFO_MAINTAIN+1,
   BN_PRIM_BWD_SCALESHIFT                 = MKL_INFO_MAINTAIN+2,
   BN_PRIM_CVT_GRADOUTPUT_BWD             = MKL_INFO_MAINTAIN+3,

   BN_BUFFER_WORKSPACE                    = MKL_INFO_MAINTAIN+4,
   BN_BUFFER_SCALESHIFT                   = MKL_INFO_MAINTAIN+5,
   BN_BUFFER_GRADOUTPUT_BWD               = MKL_INFO_MAINTAIN+6
} mkldnnBNIndex_t;

typedef enum {
   LRN_PRIM_FWD                           = MKL_INFO_MAINTAIN+0,
   LRN_PRIM_BWD                           = MKL_INFO_MAINTAIN+1,
   LRN_PRIM_CVT_GRADOUTPUT_BWD            = MKL_INFO_MAINTAIN+2,

   LRN_BUFFER_WORKSPACE                   = MKL_INFO_MAINTAIN+3,
   LRN_BUFFER_GRADOUTPUT_BWD              = MKL_INFO_MAINTAIN+4
} mkldnnLRNIndex_t;

typedef enum {
   CONCAT_PRIM_FWD                        = MKL_INFO_MAINTAIN+0,
   CONCAT_PRIM_BWD                        = MKL_INFO_MAINTAIN+1
} mkldnnConcatIndex_t;

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        fprintf(stderr,"[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    } \
} while(0)

#endif

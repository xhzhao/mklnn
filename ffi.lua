local ffi = require 'ffi'


local cdefs = [[

void MKLNN_RealSpatialConvolution_forward(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  THRealTensor *weight,
  THRealTensor *bias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);

void MKLNN_RealSpatialConvolution_bwdData(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput,
  THRealTensor *weight,
  THRealTensor *bias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);

void MKLNN_RealSpatialConvolution_bwdFilter(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THRealTensor *gradWeight,
  THRealTensor *gradBias,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  real scale,
  int group);

void MKLNN_RealThreshold_updateOutput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  real threshold);

void MKLNN_RealThreshold_updateGradInput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput);

void MKLNN_RealSpatialMaxPooling_updateOutput(
  THMKLLongTensor *primitives,
  int initOK, 
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode);

void MKLNN_RealSpatialMaxPooling_updateGradInput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput);

void MKLNN_RealSpatialAveragePooling_updateOutput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  bool count_include_pad);

void MKLNN_RealSpatialAveragePooling_updateGradInput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput);

void MKLNN_RealBatchNormalization_updateOutput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input, 
  THMKLRealTensor *output,
  THRealTensor *weight, 
  THRealTensor *bias,
  THRealTensor *running_mean, 
  THRealTensor *running_var,
  bool train, 
  double momentum, 
  double eps);

void MKLNN_RealBatchNormalization_backward(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input, 
  THMKLRealTensor *gradOutput, 
  THMKLRealTensor *gradInput,
  THRealTensor *gradWeight, 
  THRealTensor *gradBias, 
  THRealTensor *weight);

void MKLNN_RealCrossChannelLRN_updateOutput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input, 
  THMKLRealTensor *output,
  int size, 
  float alpha, 
  float beta, 
  float k);

void MKLNN_RealCrossChannelLRN_backward(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor *input, 
  THMKLRealTensor *gradOutput, 
  THMKLRealTensor *gradInput);

void MKLNN_RealConcat_updateOutput(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor **inputArray,
  THMKLRealTensor *output,
  int moduleNum);

void MKLNN_RealConcat_backward_split(
  THMKLLongTensor *primitives,
  int initOK,
  THMKLRealTensor **gradInputArray,
  THMKLRealTensor *gradOutput,
  int moduleNum);

void MKLNN_Realrandom_bernoulli(
  THRealTensor *self,
  double p);

]]

local Real2real = {
   Float='float',
   Double='double'
}


for Real, real in pairs(Real2real) do
   local type_cdefs=cdefs:gsub('Real', Real):gsub('real', real)
   ffi.cdef(type_cdefs)
end


local MKLENGINE_PATH = package.searchpath('libmklnn', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)

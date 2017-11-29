--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs NOT coming from convolution layers.
   For convolution layers, use nn.SpatialBatchNormalization.

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
        standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.BatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(N [,eps] [,momentum], false)

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentum of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--


local BN,parent = torch.class('mklnn.BatchNormalization', 'nn.Module')
local THNN = require 'nn.THNN'

local wrapper = mklnn.wrapper
local getType = mklnn.getType

BN.__version = 2

-- expected dimension of input
BN.nDim = 2

function BN:__init(nOutput, eps, momentum, affine, running_mean, running_var, weight, bias)
   parent.__init(self)
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end

   if (nOutput) then
      assert(nOutput and type(nOutput) == 'number',
             'Missing argument #1: dimensionality of input. ')
      assert(nOutput ~= 0, 'To set affine=false call BatchNormalization'
        .. '(nOutput,  eps, momentum, false) ')

      self.running_mean = torch.zeros(nOutput)
      self.running_var = torch.ones(nOutput)

      if self.affine then
         self.weight = torch.Tensor(nOutput)
         self.bias = torch.Tensor(nOutput)
         self.gradWeight = torch.Tensor(nOutput)
         self.gradBias = torch.Tensor(nOutput)
         self:reset()
      end
   else
      assert(running_mean and running_var and weight and bias, "For convertion, all parameters should be passed in")
      self.running_mean = running_mean 
      self.running_var = running_var

      if self.affine then
         self.weight = weight
         self.bias = bias
         self.gradWeight = torch.Tensor(running_mean:size())
         self.gradBias = torch.Tensor(running_mean:size())
      end
  
   end

   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1

   self.mkldnnInitOK =  false
   self.firstIteration = true

end

function BN:reset()
   if self.weight then
      self.weight:uniform()
   end
   if self.bias then
      self.bias:zero()
   end
   self.running_mean:zero()
   self.running_var:fill(1)
end

function BN:checkInputDim(input)
end

local function makeContiguous(self, input, gradOutput)
   return input, gradOutput
end

function BN:updateOutput(input)
   self:checkInputDim(input)

   if self.firstIteration then
      self.dnnPrimitives = self.dnnPrimitives and self.dnnPrimitives:zero() or torch.LongTensor(10):zero():mkl()
      self.mkldnnInitOK = false
      self.firstIteration = false 
   else
      self.mkldnnInitOK = true
   end

   self.output = self.output:mkl()
   wrapper(getType(input),'BatchNormalization_updateOutput',
      self.dnnPrimitives:cdata(),self.mkldnnInitOK,
      input:cdata(),
      self.output:cdata(),
      THNN.optionalTensor(self.weight),
      THNN.optionalTensor(self.bias),
      self.running_mean:cdata(),
      self.running_var:cdata(),
      self.train,
      self.momentum,
      self.eps)

   return self.output
end

local function backward(self, input, gradOutput, scale, gradInput, gradWeight, gradBias)
   self:checkInputDim(input)
   self:checkInputDim(gradOutput)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   self.gradInput = self.gradInput:mkl()
   scale = scale or 1
   
   if gradInput then
      wrapper(getType(input),'BatchNormalization_backward',
         self.dnnPrimitives:cdata(),self.mkldnnInitOK,
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         THNN.optionalTensor(gradWeight),
         THNN.optionalTensor(gradBias),
         THNN.optionalTensor(self.weight))
   end
   return self.gradInput
end

function BN:backward(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)
end

function BN:updateGradInput(input, gradOutput)
   return backward(self, input, gradOutput, 1, self.gradInput)
end

function BN:accGradParameters(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, nil, self.gradWeight, self.gradBias)
end

function BN:read(file, version)
   parent.read(self, file)
   if version < 2 then
      if self.running_std then
         self.running_var = self.running_std:pow(-2):add(-self.eps)
         self.running_std = nil
      end
   end
end

function BN:clearState()
   -- first 5 buffers are not present in the current implementation,
   -- but we keep them for cleaning old saved models
   nn.utils.clear(self, {
      'buffer',
      'buffer2',
      'centered',
      'std',
      'normalized',
      '_input',
      '_gradOutput',
   })
   print("================= bn")
   self.mkldnnInitOK =  false
   self.firstIteration = true
   self.dnnPrimitives = nil
   return parent.clearState(self)
end

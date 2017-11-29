local LRN, parent = torch.class('mklnn.SpatialCrossMapLRN', 'nn.Module')
local ffi = require 'ffi'

local wrapper = mklnn.wrapper
local getType = mklnn.getType

function LRN:__init(size, alpha, beta, k)
   parent.__init(self)
   self.size = size or 5
   self.alpha = alpha or 1e-4
   self.beta = beta or 0.75
   self.k = k or 1.0
   assert(self.size >= 1 and self.size <= 16, "size has to be between 1 and 16")
   assert(self.k >= 1e-5, "k has to be greater than 1e-5")
   assert(self.beta >= 0.01, "Beta has to be > 0.01")

   self.mkldnnInitOK =  false
   self.firstIteration = true
end

function LRN:updateOutput(input)
   if self.K then self.k, self.K = self.K, nil end
   if self.firstIteration then
      self.dnnPrimitives = self.dnnPrimitives and self.dnnPrimitives:zero() or torch.LongTensor(8):zero():mkl()
      self.mkldnnInitOK = false
      self.firstIteration = false
   else 
      self.mkldnnInitOK = true
   end 

   self.output = self.output:mkl()
   self.gradInput = self.gradInput:mkl()
   wrapper(getType(input),'CrossChannelLRN_updateOutput',
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOK,
      input:cdata(),
      self.output:cdata(),
      self.size,
      self.alpha,
      self.beta,
      self.k
      )
   return self.output
end

function LRN:updateGradInput(input, gradOutput)
   if not self.gradInput then return end

   wrapper(getType(input),'CrossChannelLRN_backward',
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOK,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata()
      )
   return self.gradInput
end

function LRN:write(f)
   --self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function LRN:clearState()
   print("================= lrn")
   self.dnnPrimitives = nil
   self.mkldnnInitOK =  false
   self.firstIteration = true
   return nn.Module.clearState(self)
end

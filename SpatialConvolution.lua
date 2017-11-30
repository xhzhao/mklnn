local SpatialConvolution, parent = torch.class('mklnn.SpatialConvolution', 'nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType
function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,group)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.group = group or 1
   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW/self.group)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW/self.group)
   self.gradBias = torch.Tensor(nOutputPlane)
   self:reset()

   self.mkldnnInitOK =  false
   self.firstIteration = true
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function makeContiguous(self, input, gradOutput)
   return input, gradOutput
end

function SpatialConvolution:updateOutput(input)
   if self.firstIteration then
      self.dnnPrimitives = self.dnnPrimitives and self.dnnPrimitives:zero() or torch.LongTensor(33):zero():mkl()
      self.mkldnnInitOK = false
      self.firstIteration = false
      self.output = self.output:mkl()
      self.gradInput = self.gradInput:mkl()
   else 
      self.mkldnnInitOK = true
   end 
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)
   wrapper(getType(input),'SpatialConvolution_forward',
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOK,
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,self.group
   )
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      wrapper(getType(input),'SpatialConvolution_bwdData',
         self.dnnPrimitives:cdata(),
         self.mkldnnInitOK,
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.bias:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH,self.group
         )
   return self.gradInput
   end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input, gradOutput = makeContiguous(self, input, gradOutput)
   wrapper(getType(input),'SpatialConvolution_bwdFilter',
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOK,
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale, self.group
   )
end

function SpatialConvolution:type(type,tensorCache)
   return parent.type(self,type,tensorCache)
end

function SpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

function SpatialConvolution:clearState()
   nn.utils.clear(self, '_input', '_gradOutput')
   self.mkldnnInitOK =  false
   self.firstIteration = true
   self.dnnPrimitives = nil
   return parent.clearState(self)
end


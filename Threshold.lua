local Threshold, parent = torch.class('mklnn.Threshold','nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType
function Threshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end

   self.mkldnnInitOK =  false
   self.firstIteration = true
   self:validateParameters()
end

function Threshold:updateOutput(input)
   if self.firstIteration then
      self.dnnPrimitives = self.dnnPrimitives and self.dnnPrimitives:zero() or torch.LongTensor(10):zero():mkl()
      self.mkldnnInitOK = false
      self.firstIteration = false 
      self.gradInput = self.gradInput:mkl()
      self.output = self.output:mkl()
      self:validateParameters()
   else
      self.mkldnnInitOK = true
   end

   wrapper(getType(input),'Threshold_updateOutput',
           self.dnnPrimitives:cdata(),
           self.mkldnnInitOK,
           input:cdata(),
           self.output:cdata(),
           self.threshold
          ) 
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   if self.firstIteration then
     self:validateParameters()
   end
   wrapper(getType(input),'Threshold_updateGradInput',
              self.dnnPrimitives:cdata(),
              self.mkldnnInitOK,
              input:cdata(),
              gradOutput:cdata(),
              self.gradInput:cdata()
          )
   return self.gradInput
end

function Threshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end

function Threshold:clearState()
   self.dnnPrimitives = nil
   self.mkldnnInitOK =  false
   self.firstIteration = true
   print('===============Threshold')
   return parent.clearState(self)
end



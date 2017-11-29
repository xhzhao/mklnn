local Concat, parent = torch.class('mklnn.Concat', 'nn.Container')
local ffi = require 'ffi'
local wrapper = mklnn.wrapper
local getType = mklnn.getType

function Concat:__init(dimension)
   parent.__init(self)
   self.outputSize = torch.LongStorage()
   self.dimension = dimension
   self.mkldnnInitOK =  false
   self.firstIteration = true
end

function Concat:updateOutput(input)
   self.outputSize = self.outputSize or torch.LongStorage()
   if self.firstIteration then
      self.dnnPrimitives = self.dnnPrimitives and self.dnnPrimitives:zero() or torch.LongTensor(5):zero():mkl()
      self.mkldnnInitOK = false
      self.firstIteration = false 
   else
      self.mkldnnInitOK = true
   end

   local outs_ptr = {}
   local outs = {}
   for i=1,#self.modules do
      local currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      outs[i] = currentOutput
      outs_ptr[i] = currentOutput:cdata()

      if i == 1 then
         self.outputSize:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.outputSize[self.dimension] = self.outputSize[self.dimension] + currentOutput:size(self.dimension)
      end
   end
   string_type = torch.type(outs[1])
   cdefs = string_type:gsub('torch.', 'struct TH')
   type_outs_ptr = cdefs .. "*[" .. #outs_ptr .."]"
   ffi_outs = ffi.new(type_outs_ptr, outs_ptr)

   self.output = self.output:mkl()
   self.output:resize(self.outputSize)
   wrapper(getType(self.output),
          'Concat_updateOutput',
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOK,
          ffi_outs, 
          self.output:cdata(),
          tonumber(#self.modules)
          )

   return self.output
end

function Concat:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput:mkl()
   self.gradInput:resizeAs(input)
   local gradOutput = {}
   local gradOutput_ptr = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      gradOutput[i] = gradOutputPart
      gradOutput_ptr[i] = gradOutputPart:cdata()
   end

   string_type = torch.type(gradOutput[1])
   cdefs = string_type:gsub('torch.', 'struct TH')
   type_gradOuts_ptr = cdefs .. "*[" .. #gradOutput_ptr .."]"
   ffi_gradOuts = ffi.new(type_gradOuts_ptr, gradOutput_ptr)

   wrapper(getType(gradOutput),
          'Concat_backward_split',
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOK,
          ffi_gradOuts,
          gradOutput:cdata(),
          tonumber(#self.modules)
          )
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputBuffer[i]
      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutputPart)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      self:rethrowErrors(module, i, 'accGradParameters',
          input,
          gradOutputPart,
          scale)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:backward(input, gradOutput, scale)
   self.gradInput = self.gradInput:mkl()
   self.gradInput:resizeAs(input)
   local gradOutputs = {}
   local gradOutputs_ptr = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      gradOutputs[i] = gradOutputPart
      gradOutputs_ptr[i] = gradOutputPart:cdata()
   end

   string_type = torch.type(gradOutputs[1])
   cdefs = string_type:gsub('torch.', 'struct TH')
   type_gradOuts_ptr = cdefs .. "*[" .. #gradOutputs_ptr .."]"
   ffi_gradOuts = ffi.new(type_gradOuts_ptr, gradOutputs_ptr)

   wrapper(getType(gradOutput),
          'Concat_backward_split',
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOK,
          ffi_gradOuts,
          gradOutput:cdata(),
          tonumber(#self.modules)
          )

   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputs[i]
      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutputPart)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function Concat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      self:rethrowErrors(module, i, 'accUpdateGradParameters',
          input,
          gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
          lr)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end

function Concat:clearState()
   print('===============Concat')
   self.dnnPrimitives = nil
   self.mkldnnInitOK =  false
   self.firstIteration = true
   return parent.clearState(self)
end

local I2U, parent = torch.class('mklnn.I2U','nn.Module')


function I2U:__init()
   parent.__init(self)
end

function I2U:updateOutput(input)
   if input:type() == 'torch.MKLFloatTensor' or  input:type() == 'torch.MKLDoubleTensor' then
      self.output = input:th()
      return self.output
   else
      print("Warning: I2U op forward, input is not torch.MKLFloatTensor or torch.MKLDoubleTensor")
      return input
   end
end

function I2U:updateGradInput(input, gradOutput)
   if gradOutput:type() == 'torch.FloatTensor' or  gradOutput:type() == 'torch.DoubleTensor' then
      self.gradInput = gradOutput:mkl()
      return self.gradInput
   else
      print("Warning: I2U op backward, gradOutput is not torch.FloatTensor or torch.DoubleTensor")
      return gradOutput
   end

end

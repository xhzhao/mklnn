local U2I, parent = torch.class('mklnn.U2I','nn.Module')


function U2I:__init()
   parent.__init(self)
end

function U2I:updateOutput(input)
   if input:type() == 'torch.FloatTensor' or input:type() == 'torch.DoubleTensor' then
      self.output = input:mkl()
      return self.output
   else
      print("Warning: U2I op forward, input is not torch.FloatTensor or torch.DoubleTensor")
      return input
   end
end

function U2I:updateGradInput(input, gradOutput)
   if gradOutput:type() == 'torch.MKLFloatTensor' or gradOutput:type() == 'torch.MKLDoubleTensor' then
      self.gradInput = gradOutput:th()
      return self.gradInput
   else
      print("Warning: U2I op backward, gradOutput is not torch.MKLFloatTensor or torch.MKLDoubleTensor")
      return gradOutput
   end

end

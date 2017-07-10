local LSTM, parent = torch.class('mklnn.LSTM', 'nn.Module')

function LSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self)
    self.mode = 'MKLNN_LSTM'
    self:reset()
end


function LSTM:updateOutput(input)
   print("mklnn.LSTM updateOutput")
end


function LSTM:updateGradInput(input, gradOutput)
   print("mklnn.LSTM updateGradInput")
end

function LSTM:accGradParameters(input, gradOutput, scale)
   print("mklnn.LSTM accGradParameters")
end

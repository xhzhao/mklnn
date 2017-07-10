local LSTM, parent = torch.class('mklnn.LSTMFullStep', 'nn.Module')

function LSTM:__init(stepSize, inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self)
    self.mode = 'MKLNN_LSTM'
    self:reset()
end


function LSTM:updateOutput(input)
   print("mklnn.LSTMFullStep updateOutput")
end


function LSTM:updateGradInput(input, gradOutput)
   print("mklnn.LSTMFullStep updateGradInput")
end

function LSTM:accGradParameters(input, gradOutput, scale)
   print("mklnn.LSTMFullStep accGradParameters")
end

require 'nn'
require 'mklnn'

local N, T, D, H = 64,32,500,500
local lstm = mklnn.LSTMFullStep(D, H):float()

for i=1,100 do
  local h0 = torch.randn(N, H):float()
  local c0 = torch.randn(N, H):float()
  local x  = torch.randn(T, N, D):float()

  sys.tic()
  local output_table = lstm:forward{c0, h0, x}
  print("Lua time = ",sys.toc())
end


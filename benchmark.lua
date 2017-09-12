require 'nn'
require 'mklnn'

local COUNT=10
--local N, T, D, H = 64,50,500,500
local N, T, D, H = 128,25,4096,4096
-- N: batchsize, T: time step, D: input dim, H: output dim
--   -- no layer size

local h0 = torch.randn(N, H)
local c0 = torch.randn(N, H)
local x  = torch.randn(T, N, D)

local lstm = mklnn.LSTMFullStep(D, H)


for iter=1,COUNT do 
  lstm:forward{c0, h0, x}
end

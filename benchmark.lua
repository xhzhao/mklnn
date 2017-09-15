require 'nn'
require 'mklnn'

local COUNT=5
local sizes={}
sizes[0]={16,25,512,512}
sizes[1]={32,25,512,512}
sizes[2]={64,25,512,512}
sizes[3]={128,25,512,512}

sizes[4]={16,25,1024,1024}
sizes[5]={32,25,1024,1024}
sizes[6]={64,25,1024,1024}
sizes[7]={128,25,1024,1024}

sizes[8]={16,25,2048,2048}
sizes[9]={32,25,2048,2048}
sizes[10]={64,25,2048,2048}
sizes[11]={128,25,2048,2048}

sizes[12]={16,25,4096,4096}
sizes[13]={32,25,4096,4096}
sizes[14]={64,25,4096,4096}
sizes[15]={128,25,4096,4096}

sizes[16]={64,50,500,500}

for s=0,16 do
  size = sizes[s]
  local N = size[1] 
  local T = size[2] 
  local D = size[3] 
  local H = size[4] 
  --local N, T, D, H = 128,25,4096,4096
  -- N: batchsize, T: time step, D: input dim, H: output dim
  --   -- no layer size
  
  local h0 = torch.randn(N, H):float()
  local c0 = torch.randn(N, H):float()
  local x  = torch.randn(T, N, D):float()
  
  local lstm = mklnn.LSTMFullStep(D, H):float()
  
  
  for iter=1,COUNT do 
    lstm:forward{c0, h0, x}
  end
end

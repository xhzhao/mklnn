

local LSTM, parent = torch.class('mklnn.LSTMFullStep', 'nn.Module')
local wrapper = mklnn.wrapper
local getType = mklnn.getType

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]


function LSTM:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D+H, 4 * H)
  self.weightX = self.weight[{{1,D}}]
  self.weightH = self.weight[{{D+1,D+H}}]

  self.Wx_t = torch.Tensor(4,D,H)
  self.Wh_t = torch.Tensor(4,H,H)
  self.gates_t = torch.Tensor()

  self.gradWeight = torch.Tensor(D + H, 4 * H):zero()
  self.bias = torch.Tensor(4 * H)
  self.gradBias = torch.Tensor(4 * H):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (T, N, H)
  self.gates = torch.Tensor()   -- This will be (T, N, 4H)
  self.gates_mkl = torch.Tensor()
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (1, 4H)
  self.grad_a_buffer1 = torch.Tensor() -- This will be (N, 4H)
  self.grad_a_buffer2 = torch.Tensor() -- This will be (N, 4H)

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.remember_states = false

  self.grad_c0 = torch.Tensor()
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
end


function LSTM:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  self.bias:zero()
  --self.bias[{{self.hidden_dim + 1, 2 * self.hidden_dim}}]:fill(1)
  self.weightX:normal(0, std)
  self.weightH:normal(0, std)
  return self
end


function LSTM:resetStates()
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function LSTM:_unpack_input(input)
  local c0, h0, x = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    c0, h0, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return c0, h0, x
end


function LSTM:_get_sizes(input, gradOutput)
  local c0, h0, x = self:_unpack_input(input)
  local T, N = x:size(1), x:size(2)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {T, N, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  if c0 then
    check_dims(c0, {N, H})
  end
  if gradOutput then
    check_dims(gradOutput, {T, N, H})
  end
  return T, N, D, H
end


--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, H)
--]]


function LSTM:updateOutput(input)
  --print("mklnn.LSTM updateOutput")
  self.recompute_backward = true
  local c0, h0, x = self:_unpack_input(input)
  local T, N, D, H = self:_get_sizes(input)

  self._return_grad_c0 = (c0 ~= nil)
  self._return_grad_h0 = (h0 ~= nil)
  if not c0 then
    c0 = self.c0
    if c0:nElement() == 0 or not self.remember_states then
      c0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.cell:size(1), self.cell:size(2)
      assert(prev_N == N, 'batch sizes must be constant to remember states')
      c0:copy(self.cell[{{}, prev_T}])
    end
  end
  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.output:size(1), self.output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self.output[{{}, prev_T}])
    end
  end

  local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)
  local Wx = self.weightX
  local Wh = self.weightH

  local h, c = self.output, self.cell
  h:resize(T, N, H):zero()
  c:resize(T, N, H):zero()
  local prev_h, prev_c = h0, c0
  self.gates:resize(T, N, 4 * H):zero()

  self.Wx_t[1] = Wx[{{}, {1, H}}]
  self.Wx_t[2] = Wx[{{}, {H + 1, 2 * H}}]
  self.Wx_t[3] = Wx[{{}, {2 * H + 1, 3 * H}}]
  self.Wx_t[4] = Wx[{{}, {3 * H + 1, 4 * H}}]

  self.Wh_t[1] = Wh[{{}, {1, H}}]
  self.Wh_t[2] = Wh[{{}, {H + 1, 2 * H}}]
  self.Wh_t[3] = Wh[{{}, {2 * H + 1, 3 * H}}]
  self.Wh_t[4] = Wh[{{}, {3 * H + 1, 4 * H}}]



  self.gates_mkl:resize(T, 4 * N, H):zero()
  wrapper(getType(x),'LSTMFullStep_updateOutput',
      x:cdata(),
      self.Wx_t:cdata(),
      self.Wh_t:cdata(),
      bias_expand:cdata(),
      c:cdata(),
      h:cdata(),
      c0:cdata(),
      h0:cdata(),
      self.gates_mkl:cdata())

  self.gates[{{1,T}, {1,N}, {1, H}}]           = self.gates_mkl[{{1,T},{1,N},           {1,H}}]
  self.gates[{{1,T}, {1,N}, {H + 1, 2 * H}}]   = self.gates_mkl[{{1,T},{N + 1, 2 * N},  {1,H}}]
  self.gates[{{1,T}, {1,N}, {2*H + 1, 3 * H}}] = self.gates_mkl[{{1,T},{2*N + 1, 3 * N},{1,H}}]
  self.gates[{{1,T}, {1,N}, {3*H + 1, 4 * H}}] = self.gates_mkl[{{1,T},{3*N + 1, 4 * N},{1,H}}]


--[[

  for t = 1, T do
    local cur_x = x[{t,{}}]
    local next_h = h_ori[{t, {}}]
    local next_c = c_ori[{t, {}}]

    local cur_gates = self.gates_ori[{t, {}}]
    cur_gates:addmm(bias_expand, cur_x, Wx)
    cur_gates:addmm(prev_h, Wh)
    cur_gates[{{}, {1, 3 * H}}]:sigmoid()
    cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
    local i = cur_gates[{{}, {1, H}}]
    local f = cur_gates[{{}, {H + 1, 2 * H}}]
    local o = cur_gates[{{}, {2 * H + 1, 3 * H}}]
    local g = cur_gates[{{}, {3 * H + 1, 4 * H}}]
    next_h:cmul(i, g)
    next_c:cmul(f, prev_c):add(next_h)
    next_h:tanh(next_c):cmul(o)
    prev_h, prev_c = next_h, next_c
  end


  local check_1 = torch.all(torch.lt(torch.abs(torch.add(h_ori, -h_mkl)), 1e-6))
  local check_2 = torch.all(torch.lt(torch.abs(torch.add(c_ori, -c_mkl)), 1e-6))
  local check_3 = torch.all(torch.lt(torch.abs(torch.add(self.gates_ori, -self.gates_mkl)), 1e-6))
  print("result check = ",check_1, check_2, check_3)
]]--


  local next_h = h[{T, {}}]
  local next_c = c[{T, {}}]
  return {h, c,next_h,next_c}
end


function LSTM:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  local c0, h0, x = self:_unpack_input(input)
  if not c0 then c0 = self.c0 end
  if not h0 then h0 = self.h0 end

  local grad_c0, grad_h0, grad_x = self.grad_c0, self.grad_h0, self.grad_x
  local h, c = self.output, self.cell
  local grad_h = gradOutput

  local T, N, D, H = self:_get_sizes(input, gradOutput)
  local Wx = self.weightX
  local Wh = self.weightH
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  grad_h0:resizeAs(h0):zero()
  grad_c0:resizeAs(c0):zero()
  grad_x:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local grad_next_c = self.buffer2:resizeAs(c0):zero()


  local grad_x_mkl = grad_x:clone()
  local grad_b_mkl = grad_b:clone()
  local grad_c0_mkl = grad_c0:clone()
  local grad_h0_mkl = grad_h0:clone()
  local grad_Wx_mkl = grad_Wx:clone()
  local grad_Wh_mkl = grad_Wh:clone()

  print("x size = ", x:size())
  print("self.Wx_t size = ", self.Wx_t:size())
  print("self.Wh_t size = ", self.Wh_t:size())
  print("gradOutput size = ", gradOutput:size())
  print("grad_x_mkl size = ", grad_x_mkl:size())
  print("grad_b_mkl size = ", grad_b_mkl:size())
  print("grad_c0_mkl size = ", grad_c0_mkl:size())
  print("grad_h0_mkl size = ", grad_h0_mkl:size())

  print("self.gates_mkl size = ", self.gates_mkl:size())

  self.grad_a_buffer1:resize(N, 4 * H):zero()
  self.grad_a_buffer2:resize(N, 4 * H):zero()
  wrapper(getType(x),'LSTMFullStep_updateGradInput',
      x:cdata(),
      Wx:cdata(),
      Wh:cdata(),
      gradOutput:cdata(),
      h:cdata(),
      c:cdata(),
      h0:cdata(),
      c0:cdata(),
      self.gates_mkl:cdata(),
      grad_x_mkl:cdata(),
      grad_b_mkl:cdata(),
      grad_c0_mkl:cdata(),
      grad_h0_mkl:cdata(),
      grad_Wx_mkl:cdata(),
      grad_Wh_mkl:cdata(),
      self.grad_a_buffer1:cdata(),
      self.grad_a_buffer2:cdata()
)
  


  for t = T, 1, -1 do
    local next_h, next_c = h[{t, {}}], c[{t, {}}]
    local prev_h, prev_c = nil, nil
    if t == 1 then
      prev_h, prev_c = h0, c0
    else
      prev_h, prev_c = h[{t-1, {}}], c[{t-1, {}}]
    end
    grad_next_h:add(grad_h[{t,{}}])

    local i = self.gates[{t, {}, {1, H}}]
    local f = self.gates[{t, {}, {H + 1, 2 * H}}]
    local o = self.gates[{t, {}, {2 * H + 1, 3 * H}}]
    local g = self.gates[{t, {}, {3 * H + 1, 4 * H}}]
    
    local grad_a = self.grad_a_buffer1:resize(N, 4 * H):zero()
    local grad_ai = grad_a[{{}, {1, H}}]
    local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
    local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
    
    -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
    -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
    -- to compute grad_ao; the other values can be overwritten after we compute
    -- grad_next_c
--[[
    if t==T-1 then
      print("next_c sum  = ",next_c:sum())
      print("grad_next_h sum = ",grad_next_h:sum())
      print("ot sum = ",o:sum())
      
    end
]]--
    local tanh_next_c = grad_ai:tanh(next_c)
    local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
    local my_grad_next_c = grad_ao
    my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
    grad_next_c:add(my_grad_next_c)
   
     
    -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
    -- that we can overwrite it.
    grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

    -- Use grad_ai as a temporary buffer for computing grad_ag
    local g2 = grad_ai:cmul(g, g)
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

    -- We don't need any temporary storage for these so do them last
    grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
    grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
    

    grad_x[{t, {}}]:mm(grad_a, Wx:t())
    local tmp = torch.mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[{t, {}}]:t(), grad_a)
    grad_Wh:addmm(scale, prev_h:t(), grad_a)
    local grad_a_sum = self.buffer3:resize(1, 4 * H):sum(grad_a, 1)
    grad_b:add(scale, grad_a_sum)

    grad_next_h:mm(grad_a, Wh:t())
    grad_next_c:cmul(f)
    --if t==T-1 then
      print("---------------------------------------t = ",t)
      print("grad_ao sum = ",grad_ao:sum())
      print("grad_ag sum = ",grad_ag:sum())
      print("grad_ai sum = ",grad_ai:sum())
      print("grad_af sum = ",grad_af:sum())
      print("grad_a sum = ", grad_a:sum())
      print("Wx sum = ",     Wx:sum())
      print("grad_x sum = ", grad_x[{t, {}}]:sum())
      print("grad_Wx sum = ",grad_Wx:sum())
      print("grad_Wh sum = ",grad_Wh:sum())
      print("grad_next_h sum = ",grad_next_h:sum())
      print("grad_next_c sum = ",grad_next_c:sum())
    --end
  end
  grad_h0:copy(grad_next_h)
  grad_c0:copy(grad_next_c)

  if self._return_grad_c0 and self._return_grad_h0 then
    self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
  elseif self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_x}
  else
    self.gradInput = self.grad_x
  end

  local check_1 = torch.all(torch.lt(torch.abs(torch.add(grad_c0,   -grad_c0_mkl)), 1e-6))
  local check_2 = torch.all(torch.lt(torch.abs(torch.add(grad_h0,   -grad_h0_mkl)), 1e-6))
  local check_3 = torch.all(torch.lt(torch.abs(torch.add(grad_x, -grad_x_mkl)), 1e-6))
  local check_4 = torch.all(torch.lt(torch.abs(torch.add(grad_Wx, -grad_Wx_mkl)), 1e-6))
  local check_5 = torch.all(torch.lt(torch.abs(torch.add(grad_Wh, -grad_Wh_mkl)), 1e-6))
  print("result check = ",check_1, check_2, check_3,check_4,check_5)
  print("grad_c0 = ",grad_c0:sum()," grad_c0_mkl sum = ",grad_c0_mkl:sum())
  print("grad_h0 = ",grad_h0:sum()," grad_h0_mkl sum = ",grad_h0_mkl:sum())
  print("grad_x = ",grad_x:sum()," grad_x_mkl sum = ",grad_x_mkl:sum())

  return self.gradInput
end


function LSTM:clearState()
  self.cell:set()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

  self.grad_c0:set()
  self.grad_h0:set()
  self.grad_x:set()
  self.output:set()
end


function LSTM:updateGradInput(input, gradOutput)
  print("mklnn.LSTM updateGradInput")
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end


function LSTM:accGradParameters(input, gradOutput, scale)
  print("mklnn.LSTM accGradParameters")
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end


function LSTM:__tostring__()
  local name = torch.type(self)
  local din, dout = self.input_dim, self.hidden_dim
  return string.format('%s(%d -> %d)', name, din, dout)
end



local mklOP2thOP = {}
mklOP2thOP['mklnn.SpatialConvolution']        = nn.SpatialConvolution
mklOP2thOP['nn.SpatialConvolution']           = nn.SpatialConvolution
mklOP2thOP['mklnn.SpatialMaxPooling']         = nn.SpatialMaxPooling
mklOP2thOP['nn.SpatialMaxPooling']            = nn.SpatialMaxPooling
mklOP2thOP['mklnn.SpatialAveragePooling']     = nn.SpatialAveragePooling
mklOP2thOP['nn.SpatialAveragePooling']        = nn.SpatialAveragePooling
mklOP2thOP['mklnn.SpatialCrossMapLRN']        = nn.SpatialCrossMapLRN
mklOP2thOP['nn.SpatialCrossMapLRN']           = nn.SpatialCrossMapLRN
mklOP2thOP['mklnn.ReLU']                      = nn.ReLU
mklOP2thOP['nn.ReLU']                         = nn.ReLU
mklOP2thOP['mklnn.Concat']                    = nn.Concat
mklOP2thOP['nn.Concat']                       = nn.Concat
mklOP2thOP['mklnn.Dropout']                   = nn.Dropout
mklOP2thOP['nn.Dropout']                      = nn.Dropout
mklOP2thOP['mklnn.SpatialBatchNormalization'] = nn.SpatialBatchNormalization
mklOP2thOP['nn.SpatialBatchNormalization']    = nn.SpatialBatchNormalization
mklOP2thOP['mklnn.Identity']                  = nn.Identity
mklOP2thOP['nn.Identity']                     = nn.Identity
mklOP2thOP['mklnn.CAddTable']                 = nn.CAddTable
mklOP2thOP['nn.CAddTable']                    = nn.CAddTable


local thOP2mklOP = {}
thOP2mklOP['nn.SpatialConvolution']           = mklnn.SpatialConvolution
thOP2mklOP['mklnn.SpatialConvolution']        = mklnn.SpatialConvolution
thOP2mklOP['nn.SpatialMaxPooling']            = mklnn.SpatialMaxPooling
thOP2mklOP['mklnn.SpatialMaxPooling']         = mklnn.SpatialMaxPooling
thOP2mklOP['nn.SpatialAveragePooling']        = mklnn.SpatialAveragePooling
thOP2mklOP['mklnn.SpatialAveragePooling']     = mklnn.SpatialAveragePooling
thOP2mklOP['nn.SpatialCrossMapLRN']           = mklnn.SpatialCrossMapLRN
thOP2mklOP['mklnn.SpatialCrossMapLRN']        = mklnn.SpatialCrossMapLRN
thOP2mklOP['nn.ReLU']                         = mklnn.ReLU
thOP2mklOP['mklnn.ReLU']                      = mklnn.ReLU
thOP2mklOP['nn.Concat']                       = mklnn.Concat
thOP2mklOP['mklnn.Concat']                    = mklnn.Concat
thOP2mklOP['mklnn.Dropout']                   = mklnn.Dropout
thOP2mklOP['nn.Dropout']                      = mklnn.Dropout
thOP2mklOP['mklnn.SpatialBatchNormalization'] = mklnn.SpatialBatchNormalization
thOP2mklOP['nn.SpatialBatchNormalization']    = mklnn.SpatialBatchNormalization
thOP2mklOP['mklnn.Idenity']                   = mklnn.Identity
thOP2mklOP['nn.Identity']                     = mklnn.Identity
thOP2mklOP['mklnn.CAddTable']                 = mklnn.CAddTable
thOP2mklOP['nn.CAddTable']                    = mklnn.CAddTable

--[[
NOTE:
the model won't convert to the other version when OPs of source model are same with the refered OPs you specify 
src_model:  model to be convert to the other version
th2mkl:    when th2mkl==0, the thinary OP will convert to mkldnn OP
                when th2mkl!=0, the mkldnn OP will convert to thinary OP
]]--


local convert = function(src_model, th2mkl)
  
  local cvtOp = th2mkl or 'mkl'
  if ('mkl' == th2mkl) then
    model_flag, model = nn2mklnnCvt(src_model, thOP2mklOP, false) --false: regular, true: mklnn
  elseif('nn' == th2mkl) then
    model = mklnn2nnCvt(src_model, mklOP2thOP)
  else
    print("wrong type")
    return nil
  end
  if model_flag then
    local convert_layer = mklnn.I2U()
    model:add(convert_layer)
  end
  return model
end

function nn2mklnnCvt(src_module, cvtOP, prevOPFlag)
  local dst_module
  local module_type = torch.type(src_module)
  if(module_type == 'nn.Sequential') then
    dst_module = nn.Sequential()
    for i = 1, #src_module do
      local src_layer = src_module:get(i)
      local name = src_layer.name
      local layer_type = torch.type(src_layer)
      prevOPFlag, dst_module = nn2mklnnlayerCvt(layer_type, src_layer, prevOPFlag, dst_module, cvtOP)
    end
  elseif(string.find(module_type, 'nn.ConcatTable')) then
    local dimension = src_module.dimension
    local last_op_flag = nil
    local cat_op_flag = nil
    local op_flag_table = {}
    local sub_module_table = {}
    local add_op_flag = false
    for j = 1, #src_module.modules do 
      local dnn = src_module:get(j)
      local sub_module_flag, sub_module = nn2mklnnCvt(dnn, cvtOP, prevOPFlag)
      if (nil == last_op_flag) then
        last_op_flag = sub_module_flag
        cat_op_flag = last_op_flag
      elseif (last_op_flag ~= sub_module_flag) then
        cat_op_flag = false                  -- true:mklnn 
        add_op_flag = true
      end
      table.insert(op_flag_table, sub_module_flag)
      table.insert(sub_module_table, sub_module)    
    end 
    
    if cat_op_flag then
      concat_module = mklnn.ConcatTable(dimension)
    elseif( prevOPFlag ) then
      concat_module = mklnn.ConcatTable(dimension)
    else
      concat_module = mklnn.ConcatTable(dimension)
    end
    
    for j = 1, #src_module.modules do
      local sub_module = nil
      sub_module = sub_module_table[j]
      if add_op_flag and op_flag_table[j] ~= cat_op_flag then  
          local convert_layer = mklnn.I2U()
          sub_module:add(convert_layer)
      end 
      concat_module:add(sub_module)
    end
    if dst_module then
      dst_module:add(concat_module)
    else
      dst_module = concat_module
    end
    prevOPFlag = cat_op_flag

  elseif(string.find(module_type, 'nn.Concat')) then
    local dimension = src_module.dimension
    
    local last_op_flag = nil
    local cat_op_flag = nil
    local op_flag_table = {}
    local sub_module_table = {}
    local add_op_flag = false
    for j = 1, #src_module.modules do 
      local dnn = src_module:get(j)
      local sub_module_flag, sub_module = nn2mklnnCvt(dnn, cvtOP, prevOPFlag)
      if (nil == last_op_flag) then
        last_op_flag = sub_module_flag
        cat_op_flag = last_op_flag
      elseif (last_op_flag ~= sub_module_flag) then
        cat_op_flag = false                  -- true:mklnn 
        add_op_flag = true
      end
      table.insert(op_flag_table, sub_module_flag)
      table.insert(sub_module_table, sub_module)    
    end 
    
    if cat_op_flag then
      concat_module = mklnn.Concat(dimension)
    elseif( prevOPFlag ) then
      concat_module = mklnn.Concat2(dimension)
    else
      concat_module = nn.Concat(dimension)
    end
    
    for j = 1, #src_module.modules do
      local sub_module = nil
      sub_module = sub_module_table[j]
      if add_op_flag and op_flag_table[j] ~= cat_op_flag then  
          local convert_layer = mklnn.I2U()
          sub_module:add(convert_layer)
      end 
      concat_module:add(sub_module)
    end
    if dst_module then
      dst_module:add(concat_module)
    else
      dst_module = concat_module
    end

    prevOPFlag = cat_op_flag
  else 
    local src_layer = src_module
    local name = src_layer.name
    local layer_type = torch.type(src_layer)
    if cvtOP[layer_type] then
      prevOPFlag, dst_module = nn2mklnnlayerCvt(layer_type, src_layer, prevOPFlag, dst_module, cvtOP)
    else
      if prevOPFlag then
        dst_module = nn.Sequential()
        local convert_layer = mklnn.I2U()
        dst_module:add(convert_layer)
        dst_module:add(src_layer)
      else
        dst_module = src_layer:clone()
      end
      prevOPFlag = false
    end
  end
  return prevOPFlag, dst_module 
end

function mklnn2nnCvt(src_module, cvtOP)
  local dst_module
  local module_type = torch.type(src_module)
  if(module_type == 'nn.Sequential') then
    dst_module = nn.Sequential()
    for i = 1, #src_module do
      local src_layer = src_module:get(i)
      local layer_type = torch.type(src_layer)
      dst_module = mklnn2nnlayerCvt(layer_type, src_layer, dst_module, cvtOP)
    end
  elseif(string.find(module_type, 'mklnn.ConcatTable')) then
    local dimension = src_module.dimension
    local sub_module_table = {}
    for j = 1, #src_module.modules do
      local dnn = src_module:get(j)
      local sub_module = mklnn2nnCvt(dnn, cvtOP)
      table.insert(sub_module_table, sub_module) 
    end
    
    concat_module = nn.ConcatTable(dimension)
    for j = 1, #src_module.modules do
      local sub_module = nil
      sub_module = sub_module_table[j]
      concat_module:add(sub_module)
    end
    
    if dst_module then
      dst_module:add(concat_module)
    else
      dst_module = concat_module
    end
  
  elseif(string.find(module_type, 'mklnn.Concat')) then
    local dimension = src_module.dimension
    local sub_module_table = {}
    for j = 1, src_module:size() do
      local dnn = src_module:get(j)
      local sub_module = mklnn2nnCvt(dnn, cvtOP)
      table.insert(sub_module_table, sub_module)
    end
    concat_module = nn.Concat(dimension)
    for j = 1, src_module:size() do
      local sub_module = nil
      sub_module = sub_module_table[j]
      concat_module:add(sub_module)
    end
    if dst_module then
      dst_module:add(concat_module)
    else
      dst_module = concat_module
    end
  else
    local src_layer = src_module
    local name = src_layer.name
    local layer_type = torch.type(src_layer)
    if cvtOP[layer_type] then
      dst_module = mklnn2nnlayerCvt(layer_type, src_layer, dst_module, cvtOP)
    else
      dst_module:add(src_layer)
    end
  end
  return dst_module
end

function nn2mklnnlayerCvt(layer_type, src_layer, prevOPFlag, dst_module, cvtOP)
  if(string.find(layer_type, 'nn.SpatialConvolution')) then
    if not prevOPFlag then
      local convert_layer = mklnn.U2I()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    local nInputPlane,nOutputPlane = src_layer.nInputPlane, src_layer.nOutputPlane
    local kW,kH = src_layer.kW, src_layer.kH
    local dW,dH = src_layer.dW, src_layer.dH
    local padW,padH = src_layer.padW, src_layer.padH
    local dst_layer = cvtOP[layer_type](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    dst_layer.weight:copy(src_layer.weight)
    dst_layer.bias:copy(src_layer.bias)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end
    prevOPFlag = true

  elseif(string.find(layer_type, 'nn.SpatialMaxPooling')) then
    if not prevOPFlag then
      local convert_layer = mklnn.U2I()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    local kW,kH = src_layer.kW, src_layer.kH
    local dW,dH = src_layer.dW, src_layer.dH
    local padW,padH = src_layer.padW, src_layer.padH
    local ceil_mode = src_layer.ceil_mode
    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
    if(ceil_mode) then
      dst_layer:ceil()
    end
    dst_module:add(dst_layer)
    prevOPFlag = true
  
  elseif(string.find(layer_type, 'nn.SpatialAveragePooling')) then
    if not prevOPFlag then
      local convert_layer = mklnn.U2I()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    local kW,kH = src_layer.kW, src_layer.kH
    local dW,dH = src_layer.dW, src_layer.dH
    local padW,padH = src_layer.padW, src_layer.padH
    ceil_mode = src_layer.ceil_mode
    count_include_pad = src_layer.count_include_pad
    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
    if(ceil_mode) then
      dst_layer:ceil()
    end
    if(not count_include_pad) then
      dst_layer:setCountExcludePad()
    end
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end
    prevOPFlag = true
  
  elseif(string.find(layer_type, 'nn.SpatialCrossMapLRN')) then
    if not prevOPFlag then
      local convert_layer = mklnn.U2I()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    local size = src_layer.size
    local alpha, beta = src_layer.alpha, src_layer.bata
    local k = src_layer.k
    local dst_layer = cvtOP[layer_type](size, alpha, beta, k)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = sdt_layer
    end
    prevOPFlag = true
  
  elseif(string.find(layer_type, 'nn.SpatialBatchNormalization')) then
    if not prevOPFlag then
      local convert_layer = mklnn.U2I()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    local affine = src_layer.affine
    local eps = src_layer.eps
    local momentum = src_layer.momentum
    local running_mean, running_var = src_layer.running_mean, src_layer.running_var
    local weight, bias = src_layer.weight, src_layer.bias
    local dst_layer = cvtOP[layer_type](nil, eps, momentum, affine, running_mean, running_var, weight, bias)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end
    prevOPFlag = true
  
  elseif(string.find(layer_type, 'nn.ReLU')) then
    local dst_layer = src_layer
    if prevOPFlag then
      local ip = src_layer.inplace
      dst_layer = cvtOP[layer_type](ip)
      prevOPFlag = true
    end
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'nn.CAddTable')) then
    local dst_layer = cvtOP[layer_type](src_layer.inplace)
    if dst_module then
      dst_module:add(dst_layer)
    else 
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'nn.Dropout')) then
    local ip = src_layer.inplace
    local p = src_layer.p
    local train = src_layer.train
    local stochastic_inference = src_layer.stochasticInference
    local v1 = not src_layer.v2
    local dst_layer = cvtOP[layer_type](p, v1, ip, stochastic_inference)
    if prevOPFlag then
      local convert_layer = mklnn.I2U()
      prevOPFlag = false
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'nn.Identity')) then
    local dst_layer = cvtOP[layer_type]()
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end
  
  elseif(string.find(layer_type, 'nn.Concat') or string.find(layer_type, 'nn.Sequential')) then
    local model_flag, sub_module = nn2mklnnCvt(src_layer, cvtOP, prevOPFlag)
    if dst_module then
      dst_module:add(sub_module)
    else
      dst_module = sub_module
    end
    prevOPFlag = model_flag
  
  else
    if prevOPFlag then
      local convert_layer = mklnn.I2U()
      if dst_module then
        dst_module:add(convert_layer)
      else
        dst_module = convert_layer
      end
    end
    if dst_module then
      dst_module:add(src_layer)
    else 
      dst_module = src_layer
    end
    prevOPFlag = false
  end
  return prevOPFlag, dst_module
end

 

function mklnn2nnlayerCvt(layer_type, src_layer, dst_module, cvtOP)
  if(string.find(layer_type, 'mklnn.SpatialConvolution')) then
    local nInputPlane, nOutputPlane = src_layer.nInputPlane, src_layer.nOutputPlane
    local kW, kH = src_layer.kW, src_layer.kH
    local dW, dH = src_layer.dW, src_layer.dH
    local padW, padH = src_layer.padW, src_layer.padH
    local dst_layer = cvtOP[layer_type](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    dst_layer.weight:copy(src_layer.weight)
    dst_layer.bias:copy(src_layer.bias)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.SpatialMaxPooling')) then
    local kW, kH = src_layer.kW, src_layer.kH
    local dW, dH = src_layer.dW, src_layer.dH
    local padW, padH = src_layer.padW, src_layer.padH
    local ceil_mode = src_layer.ceil_mode
    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
    if(ceil_mode) then
      dst_layer:ceil()
    end
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.SpatialAveragePooling')) then
    local kW, kH = src_layer.kW, src_layer.kH
    local dW, dH = src_layer.dW, src_layer.dH
    local padW, padH = src_layer.padW, src_layer.padH
    ceil_mode = src_layer.ceil_mode
    count_include_pad = src_layer.count_include_pad
    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
    if(ceil_mode) then
      dst_layer:ceil()
    end
    if(count_include_pad) then
      dst_layer:setCountIncludePad()
    end
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end
  
  elseif(string.find(layer_type, 'mklnn.SpatialCrossMapLRN')) then
    local size = src_layer.size
    local alpha, beta = src_layer.alpha, src_layer.beta
    local k = src_layer.k
    local dst_layer = cvtOP[layer_type](size, alpha, beta, k)
    if dst_module then
      dst_module:add(dst_layer)
    else 
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.SpatialBatchNormalization')) then
    local affine = src_layer.affine
    local eps = src_layer.eps
    local momentum = src_layer.momentum
    local running_mean = src_layer.running_mean
    local nOutput = running_mean:size()[1]
    local dst_layer = cvtOP[layer_type](nOutput, eps, momentum, affine)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.ReLU')) then
    local ip = src_layer.inplace
    local dst_layer = cvtOP[layer_type](ip)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.CAddTable')) then
    local dst_layer = cvtOP[layer_type](src_layer.inplace)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.Dropout')) then
    local ip = src_layer.inplace
    local p = src_layer.p
    local train = src_layer.train
    local stochastic_inference = src_layer.stochasticInference
    local v1 = not src_layer.v2
    local dst_layer = cvtOP[layer_type](p, v1, ip, stochastic_inference)
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.Identity'))  then
    local dst_layer = cvtOP[layer_type]()
    if dst_module then
      dst_module:add(dst_layer)
    else
      dst_module = dst_layer
    end

  elseif(string.find(layer_type, 'mklnn.Concat') or string.find(layer_type, 'nn.Sequential')) then
    local sub_module = mklnn2nnCvt(src_layer, cvtOP)
    if dst_module then
      dst_module:add(sub_module)
    else
      dst_module = sub_module
    end

  elseif(string.find(layer_type, 'mklnn.I2U') or string.find(layer_type, 'mklnn.U2I')) then
    local useless = 1
  
  else
    if dst_module then
      dst_module:add(src_layer)
    else
      dst_module = src_layer
    end
  end
  return dst_module
end
    
return convert

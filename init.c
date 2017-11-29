#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stddef.h>
#include <omp.h>
#include <sys/time.h>

#include "luaT.h"
#include "THStorage.h"
#include "THTensor.h"

#include "src/mkl_cat.h"
#include "src/Random.h"
#include "src/MKLDNN.h"
#include "MKLTensor.h"

#include "src/Random.c"



#define torch_mkl_(NAME)    TH_CONCAT_4(torch_MKL, Real, Tensor_, NAME)             
#define TH_MKL_(NAME)       TH_CONCAT_4(THMKL, Real, Tensor, NAME)                                      
#define torch_mkl_tensor    TH_CONCAT_STRING_4(torch., MKL, Real, Tensor)

#define THMKLTensor         TH_CONCAT_3(THMKL, Real, Tensor)
#define MKLNN_(NAME)        TH_CONCAT_3(MKLNN_, Real, NAME)   
#define MKLDNN_(NAME)       TH_CONCAT_3(NAME, _, BIT)   


#include "generateFloatTypes.h"
#include "src/SpatialConvolution.c"
#include "generateFloatTypes.h"
#include "src/Threshold.c"
#include "generateFloatTypes.h"
#include "src/SpatialMaxPooling.c"
#include "generateFloatTypes.h"
#include "src/SpatialAveragePooling.c"
#include "generateFloatTypes.h"
#include "src/BatchNormalization.c"
#include "generateFloatTypes.h"
#include "src/SpatialCrossMapLRN.c"
#include "generateFloatTypes.h"
#include "src/Concat.c"
#include "generateFloatTypes.h"

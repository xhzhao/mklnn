#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define real float
#define accreal double
#define Real Float
#define BIT F32
#define FLOAT_TYPE 0
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef BIT
#undef FLOAT_TYPE 

#define real double
#define accreal double
#define Real Double
#define FLOAT_TYPE 1
#define BIT F64
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef BIT
#undef FLOAT_TYPE 

#undef TH_GENERIC_FILE

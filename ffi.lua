local ffi = require 'ffi'


ffi.cdef[[

typedef struct THMklFloatStorage
{
    float *data;
    long size;
    int refcount;
    char flag;
    //THAllocator *allocator;
    void *allocatorContext;
} THMklFloatStorage;



typedef struct THMklFloatTensor
{
    long *size;
    long *stride;
    int nDimension;
    
    THMklFloatStorage *storage;
    long storageOffset;
    int refcount;

    char flag;
    long mkldnnLayout;
} THMklFloatTensor;


void THNN_FloatMKLDNN_ConvertLayoutBackToNCHW(
          THFloatTensor * input,
          THLongTensor *primitives,
          int i,
          int initOk
        );
]]


local MKLENGINE_PATH = package.searchpath('libmklEngine', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)


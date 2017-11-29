#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/Random.c"
#else

#define min(x,y) (x<y?x:y)

void MKLNN_(random_bernoulli)(
  THTensor *self,
  double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");

  struct timeval start;
  gettimeofday(&start,NULL);
  long seed = start.tv_sec * 1000 + (double)start.tv_usec/1000;

  int n = THTensor_(nElement)(self);
  real *r = THTensor_(data)(self);

  RNG rng = RNGInit(seed);
  unsigned long seedNew = RandInt(&rng);

  int nthr = omp_get_max_threads();
  int *tmp = (int*)malloc(n*sizeof(int));
  #pragma omp parallel num_threads(nthr) 
  {
    int i;
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int seg_offset = ithr * avg_amount;
    const int seg_last_index_tmp = seg_offset + avg_amount;
    const int seg_last_index = seg_last_index_tmp <= n ? seg_last_index_tmp:n;
    const int seg_amount = seg_last_index - seg_offset;
         
    if (seg_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seedNew);
      vslSkipAheadStream(stream, seg_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, seg_amount, tmp+seg_offset, p);
      vslDeleteStream(&stream);
      for(i = seg_offset; i < seg_last_index; i++) {
        r[i]=tmp[i];
      }
    }
  }
  free(tmp);
}

#endif

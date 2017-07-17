#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/LSTMFullStep.c"
#else

#include <math.h>
#define LOG 0
#define BATCH_GEMM 0

//gate: 0(it), 1(ft), 2(ot), 3(gt)
static MKLNN_(LSTMFullStep_BatchGemmCrossStep)(
  int gate,
  real * x,
  real * WX,
  real * gates,
  int T,
  int N,
  int D,
  int H
)
{
#if LOG
   printf("LSTMFullStep_BatchGemmCrossStep start, x = 0x%x, WX = 0x%x, output = 0x%x, T = %d, N = %d, D = %d, H = %d\n", x, WX, gates, T, N, D, H);
#endif

   int t = 0;
   int m = N;
   int n = H;
   int k = D;

#if BATCH_GEMM
   real ** A = (real **)malloc(T*sizeof(real*));
   real ** B = (real **)malloc(T*sizeof(real*));
   real ** C = (real **)malloc(T*sizeof(real*));
   for(t = 0; t < T; t++)
   {
      A[t] = x + t * N * D;
      B[t] = WX + gate * D * H;
      C[t] = gates + t * N * 4 * H ;
   }
   CBLAS_TRANSPOSE    transA_g[1] = {CblasNoTrans};
   CBLAS_TRANSPOSE    transB_g[1] = {CblasNoTrans}; 
   int m_g[1] = {m};
   int n_g[1] = {n};
   int k_g[1] = {k};
   int lda_g[1] = {k};
   int ldb_g[1] = {n};
   int ldc_g[1] = {n};
   real alpha_g[1] = {1.0};
   real beta_g[1] = {1.0};
   int size_per_group[1] = {T};
   cblas_sgemm_batch(CblasRowMajor, transA_g, transB_g, m_g, n_g, k_g, alpha_g, A, lda_g, B, ldb_g, beta_g, C, ldc_g, 1, size_per_group);

#else

   for(t = 0; t < T; t++)
   {
      real * a = x + t * N * D;
      real * b = WX + gate * D * H;
      real * c = gates + t * N * 4 * H ;
     
      if(sizeof(real) == sizeof(float))
      {
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 1.0, c, n);
      }
/*
      if(t == 0){

      int i = 0;real tmp = 0;
      for(i=0; i < N*D; i++)
      {
         tmp+= a[i];
      }
      printf("t=1, x sum = %.4f \n", tmp);
      tmp = 0;
      for(i=0; i < D*H; i++)
      {
         tmp += b[i];
      }
      printf("t=1, WX sum = %.4f \n",tmp);
      tmp = 0;
      if(t==0)
      for(i=0; i < N*H; i++)
      {
         tmp += c[i];
      }
      printf("t=1, xi * Wx = %.4f \n",tmp);
      }
*/
      
   }
#endif

}

static MKLNN_(LSTMFullStep_BatchGemmStepInside)(
  int t,
  real * prev_h,
  real * WH,
  real * gates,
  int T,
  int N,
  int D,
  int H
)
{
#if LOG
   printf("LSTMFullStep_BatchGemmStepInside start, prev_h = 0x%x, WH = 0x%x, gates = 0x%x, T = %d, N = %d, D = %d, H = %d\n", prev_h, WH, gates, T, N, D, H);
#endif

   int i = 0;
   int m = N;
   int n = H;
   int k = H;
#if BATCH_GEMM
   real ** A = (real **)malloc(4*sizeof(real*));
   real ** B = (real **)malloc(4*sizeof(real*));
   real ** C = (real **)malloc(4*sizeof(real*));
   for(i = 0; i < 4; i++)
   {
      A[i] = prev_h;
      B[i] = WH + i * H * H;
      C[i] = gates + i * N * H;
   }
   CBLAS_TRANSPOSE    transA_g[1] = {CblasNoTrans};
   CBLAS_TRANSPOSE    transB_g[1] = {CblasNoTrans};
   int m_g[1] = {m};
   int n_g[1] = {n};
   int k_g[1] = {k};
   int lda_g[1] = {k};
   int ldb_g[1] = {n};
   int ldc_g[1] = {n};
   real alpha_g[1] = {1.0};
   real beta_g[1] = {1.0};
   int size_per_group[1] = {4};
   cblas_sgemm_batch(CblasRowMajor, transA_g, transB_g, m_g, n_g, k_g, alpha_g, A, lda_g, B, ldb_g, beta_g, C, ldc_g, 1, size_per_group);

#else
   for(i = 0; i < 4; i++)
   {
      real * a =  prev_h;
      real * b =  WH + i * H * H;
      real * c = gates + i * N * H;

      if(sizeof(real) == sizeof(float))
      {
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 1.0, c, n);
      }
/*
      if(t == 0){

      int j = 0;real tmp = 0;
      for(j=0; j < m*k; j++)
      {
         tmp+= a[j];
      }
      printf("t=1, h sum = %.4f \n", tmp);
      tmp = 0;
      for(j=0; j < k*n; j++)
      {
         tmp += b[j];
      }
      printf("t=1, WH sum = %.4f \n",tmp);
      tmp = 0;
      for(j=0; j < m*n; j++)
      {
         tmp += c[j];
      }
      printf("t=1, h * WH = %.4f \n",tmp);
      }
*/

   }
#endif

}




// input  :  T, N, D
// WX size:     D, 4H
// WH size:     H, 4H
// bias size:   N, 4H
// h  size:  T, N, H
// c  size:  T, N, H
// c0 size:     N, H
// h0 size:     N, H

void MKLNN_(LSTMFullStep_updateOutput)(
  THTensor * x,  
  THTensor * WX,
  THTensor * WH,
  THTensor * Bias,
  THTensor * c,
  THTensor * h,
  THTensor * c0,
  THTensor * h0,
  THTensor * gates)
{

   //get size: T, N, D

   int T = x->size[0];
   int N = x->size[1];
   int D = x->size[2];
   int H = h->size[2];
#if LOG
   printf("LSTMFullStep_updateOutput start\n");
   printf("T = %d, N = %d, D = %d, H = %d \n", T, N, D, H);
#endif

   //create 4 buffer to save it, ft, ot, gt
/*   real * it = malloc(T * N * H * sizeof(real));
   real * ft = malloc(T * N * H * sizeof(real));
   real * ot = malloc(T * N * H * sizeof(real));
   real * gt = malloc(T * N * H * sizeof(real));
*/
   gates = THTensor_(newContiguous)(gates);

   real * it = THTensor_(data)(gates);
   real * ft = it +     N * H;
   real * ot = it + 2 * N * H;
   real * gt = it + 3 * N * H;

   //copy bias to it,ft,ot,gt
   Bias = THTensor_(newContiguous)(Bias);
   real * bias = THTensor_(data)(Bias);
   memcpy(it, bias, N * H * sizeof(real));
   memcpy(ft, bias, N * H * sizeof(real));
   memcpy(ot, bias, N * H * sizeof(real));
   memcpy(gt, bias, N * H * sizeof(real));

/*
   int i =0;
   real tmp = 0;
   for(i=0; i<4 * N * H; i++)
   {
      tmp += it[i];
   }
   printf("bias check, sum = %.4f \n", tmp);
*/

   //1. 4 batch gemm cross step size, Xt + WXi + bi, save result to it,ft,ot,gt
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(0, THTensor_(data)(x), THTensor_(data)(WX), it, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(1, THTensor_(data)(x), THTensor_(data)(WX), ft, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(2, THTensor_(data)(x), THTensor_(data)(WX), ot, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(3, THTensor_(data)(x), THTensor_(data)(WX), gt, T, N, D, H); 

   int t = 0;
   real * prev_c = THTensor_(data)(c0);
   real * prev_h = THTensor_(data)(h0);
   for(t =0; t < T; t++)
   {

      real * next_c = THTensor_(data)(c) + t * N * H;
      real * next_h = THTensor_(data)(h) + t * N * H;
      //2. batch gemm in one step
      // gates = prev_h * WH
      MKLNN_(LSTMFullStep_BatchGemmStepInside)(t, prev_h, THTensor_(data)(WH), it, T, N, D, H) ;
      //3. Sigmoid on it,ft,ot, Tanh on gt, size = N * H * 3
      int j = 0;
      for(j = 0; j < N*H; j++)
      {
         it[j] = 1 /(1 + exp(-it[j]));
         ft[j] = 1 /(1 + exp(-ft[j]));
         ot[j] = 1 /(1 + exp(-ot[j]));
         gt[j] = tanh( gt[j] );
      }

      //4. ct, ht update
      for(j = 0; j < N*H; j++)
      {
         next_c[j] = ft[j] * prev_c[j] + it[j] * gt[j];
         next_h[j] = ot[j] * tanh(next_c[j]);
      }

      it = it +  N * 4 * H;
      ft = ft +  N * 4 * H;
      ot = ot +  N * 4 * H;
      gt = gt +  N * 4 * H;
#if 1
      prev_h = next_h;
      prev_c = next_c;
#endif
   }
}



#endif

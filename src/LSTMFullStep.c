#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/LSTMFullStep.c"
#else

#include <math.h>
#define TNUM 1
#define LOG 0
#define BATCH_GEMM 1
#define PROFILE 0
#define getTime(start,end) ((double)(end.tv_sec-start.tv_sec)*1000 + (double)(end.tv_usec-start.tv_usec)/1000)

static MKLNN_(LSTMFullStep_PrintSum)(
  real * x,
  int * len,
  char * str
)
{
   real sum = 0;
   for(int i = 0;i < len; i++)
   {
      sum += x[i];
   }
   printf("%s = %.4f\n",str,sum);
}

// x[i] = x[i] + y[i]
static MKLNN_(LSTMFullStep_Add)(
  real * x,
  real * y,
  int * len
)
{
   for(int i = 0;i < len; i++)
   {
      x[i] = x[i] + y[i];
   }
}


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
   if(sizeof(real) == sizeof(float))
   {
      cblas_sgemm_batch(CblasRowMajor, transA_g, transB_g, m_g, n_g, k_g, alpha_g, A, lda_g, B, ldb_g, beta_g, C, ldc_g, 1, size_per_group);
   }
   else if(sizeof(real) == sizeof(double))
   {
      cblas_dgemm_batch(CblasRowMajor, transA_g, transB_g, m_g, n_g, k_g, alpha_g, A, lda_g, B, ldb_g, beta_g, C, ldc_g, 1, size_per_group);
   }
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
      else if(sizeof(real) == sizeof(double))
      {
         //printf("double gemm1\n");
         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 1.0, c, n);
      }
      
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
#if 0
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
      else if(sizeof(real) == sizeof(double))
      {
         //printf("double gemm2\n");
         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 1.0, c, n);
      }

   }
#endif

}




// input  :  T, N, D
// WX size:  4, D, H
// WH size:  4, H, H
// bias size:   N, 4H
// h  size:  T, N, H
// c  size:  T, N, H
// c0 size:     N, H
// h0 size:     N, H
// gatesize: T,4N, H
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
#if PROFILE
   struct timeval start,mid1,mid2, end;
   gettimeofday(&start,NULL);
#endif
   //get size: T, N, D

   int T = x->size[0];
   int N = x->size[1];
   int D = x->size[2];
   int H = h->size[2];
#if LOG
   printf("LSTMFullStep_updateOutput start\n");
   printf("T = %d, N = %d, D = %d, H = %d \n", T, N, D, H);

#endif
   gates = THTensor_(newContiguous)(gates);

   real * it = THTensor_(data)(gates);
   real * ft = it +     N * H;
   real * ot = it + 2 * N * H;
   real * gt = it + 3 * N * H;

   //copy bias to it,ft,ot,gt
   Bias = THTensor_(newContiguous)(Bias);
   real * bias = THTensor_(data)(Bias);
   int i =0;
   for(i=0; i< T; i++)
   {
      memcpy(it + i*4*N*H, bias, N * H * sizeof(real));
      memcpy(ft + i*4*N*H, bias, N * H * sizeof(real));
      memcpy(ot + i*4*N*H, bias, N * H * sizeof(real));
      memcpy(gt + i*4*N*H, bias, N * H * sizeof(real));
   }
#if PROFILE
   gettimeofday(&mid1,NULL);
#endif


   //1. 4 batch gemm cross step size, Xt + WXi + bi, save result to it,ft,ot,gt
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(0, THTensor_(data)(x), THTensor_(data)(WX), it, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(1, THTensor_(data)(x), THTensor_(data)(WX), ft, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(2, THTensor_(data)(x), THTensor_(data)(WX), ot, T, N, D, H); 
   MKLNN_(LSTMFullStep_BatchGemmCrossStep)(3, THTensor_(data)(x), THTensor_(data)(WX), gt, T, N, D, H); 
#if PROFILE
   gettimeofday(&mid2,NULL);
   double gemm2_time = 0;
#endif

   int t = 0;
   real * prev_c = THTensor_(data)(c0);
   real * prev_h = THTensor_(data)(h0);
   for(t =0; t < T; t++)
   {

      real * next_c = THTensor_(data)(c) + t * N * H;
      real * next_h = THTensor_(data)(h) + t * N * H;
      //2. batch gemm in one step
      // gates = prev_h * WH
#if PROFILE
      struct timeval tmp1,tmp2;
      gettimeofday(&tmp1,NULL);
      MKLNN_(LSTMFullStep_BatchGemmStepInside)(t, prev_h, THTensor_(data)(WH), it, T, N, D, H) ;
      gettimeofday(&tmp2,NULL);
      gemm2_time += getTime(tmp1, tmp2);
#else
      MKLNN_(LSTMFullStep_BatchGemmStepInside)(t, prev_h, THTensor_(data)(WH), it, T, N, D, H) ;
#endif
      //3. Sigmoid on it,ft,ot, Tanh on gt, size = N * H * 3

      #pragma omp parallel num_threads(TNUM)
      {
         int tid = omp_get_thread_num();
         int j = 0;
         int block_num = (N*H)/TNUM;
         #pragma ivdep
         for(j = tid * block_num; j < (tid +1)*block_num; j++)
         {
            it[j] = 1 /(1 + exp(-it[j]));
            ft[j] = 1 /(1 + exp(-ft[j]));
            ot[j] = 1 /(1 + exp(-ot[j]));
            gt[j] = tanh( gt[j] );

         //4. ct, ht update
            next_c[j] = ft[j] * prev_c[j] + it[j] * gt[j];
            next_h[j] = ot[j] * tanh(next_c[j]);
         }
      }

      it = it +  N * 4 * H;
      ft = ft +  N * 4 * H;
      ot = ot +  N * 4 * H;
      gt = gt +  N * 4 * H;
      prev_h = next_h;
      prev_c = next_c;
   }
  
#if PROFILE
   gettimeofday(&end,NULL);
   printf("LSTM C profile, total = %.4f, init = %.4f, batchGEMM = %.4f, GEMM2 = %.4f, else = %.4f\n",getTime(start,end), getTime(start,mid1), getTime(mid1,mid2), gemm2_time, getTime(mid2,end)-gemm2_time);
#endif
   
}

// grad_gate_o  = grad_next_h * Tanh(gate_c)
// grad_next_c += grad_next_h * (1 - gate_c * gate_c) * gate_o
// len = N*H
static MKLNN_(LSTMFullStep_bprob_gateCO)(
  real * gate_c,
  real * grad_next_c,
  real * gate_o,
  real * grad_gate_o,
  real * grad_next_h,
  int len
)
{
   for(int i = 0; i < len; i++)
   {
      real tanh_c = tanh(gate_c[i]);
      grad_gate_o[i] = grad_next_h[i] * tanh_c;
      grad_next_c[i] += grad_next_h[i] * (1 - tanh_c * tanh_c) * gate_o[i];
   }
}

// grad_it = grad_next_c * gt
// grad_ft = grad_next_c * C(t-1)
// grad_gt = grad_next_c * it
// len = N*H
static MKLNN_(LSTMFullStep_bprob_gateIFG)(
  real * it,
  real * ft,
  real * gt,
  real * ct_1,
  real * grad_it,
  real * grad_ft,
  real * grad_gt,
  real * grad_next_c,
  int len
)
{
   for(int i=0; i < len; i++)
   {
      real grad_ci = grad_next_c[i];
      grad_it[i] = grad_ci * gt[i];
      grad_ft[i] = grad_ci * ct_1[i];
      grad_gt[i] = grad_ci * it[i];
   }
}

// it = sigmoid(x)
// ft = sigmoid(x)
// ot = sigmoid(x)
// gt = sigmoid(x)
static MKLNN_(LSTMFullStep_bprob_activation)(
  real * it,
  real * ft,
  real * ot,
  real * gt,
  real * grad_it,
  real * grad_ft,
  real * grad_ot,
  real * grad_gt,
  real * grad_next_c,
  int len)
{
   for(int i=0;i<len; i++)
   {
      real it_i = it[i];
      real ft_i = ft[i];
      real ot_i = ot[i];
      real gt_i = gt[i];
      grad_it[i] = it_i * (1-it_i) * grad_it[i];
      grad_ft[i] = ft_i * (1-ft_i) * grad_ft[i];
      grad_ot[i] = ot_i * (1-ot_i) * grad_ot[i];
      grad_gt[i] = (1-gt_i * gt_i) * grad_gt[i];
   }
}

// grad_a : 4N *  H
// grad_a2:  N * 4H
static MKLNN_(LSTMFullStep_bprob_transpose)(
  real * grad_a,
  real * grad_a2,
  int N,
  int H
)
{
   for(int p=0; p < 4; p++)
   {
      real * src = grad_a + p * N*H;
      real * dst = grad_a2 + p * H;
      int i = 0;
      int j = 0;
      for(i=0; i<N; i++)
         for(j=0;j<H;j++)
         {
            dst[i*4*H+j] = src[i*H+j];
         }
   }
}


// xt:     N * D
// prev_h: N * H
// WX:     D * 4H
// WH:     H * 4H
// grad_a: N * 4H
// grad_x: N * D
// grad_WX:D * 4H
// grad_WH:H * 4H
// grad_next_h: N * H
// scale 

// grad_x= grad_a * WX:trans()
// WX -= scale * ( Xt    :trans() * grad_a)
// WH -= scale * ( prev_h:trans() * grad_a)
// grad_next_h = grad_a * WH:trans()

static MKLNN_(LSTMFullStep_bprob_linear)(
  real * xt,
  real * prev_h,
  real * WX,
  real * WH,
  real * grad_a,
  real * grad_x,
  real * grad_WX,
  real * grad_WH,
  real * grad_next_h,
  real  scale,
  int T,
  int N,
  int D,
  int H)
{
    int m,n,k;
    real * a = NULL;
    real * b = NULL;
    real * c = NULL;
    // grad_x = grad_a * WX:trans()
    m = N;
    n = D;
    k = 4*H;
    a = grad_a;
    b = WX;
    c = grad_x;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, a, k, b, k, 0, c, n);

    // WX -= scale * ( Xt    :trans() * grad_a)
    m = D;
    n = 4*H;
    k = N;
    a = xt;
    b = grad_a;
    c = grad_WX;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, n, 1, c, n);

    // WH -= scale * ( prev_h:trans() * grad_a)
    m = H;
    n = 4*H;
    k = N;
    a = prev_h;
    b = grad_a;
    c = grad_WH;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, n, 1, c, n);

    // grad_next_h = grad_a * WH:trans()
    m = N;
    n = H;
    k = 4*H;
    a = grad_a;
    b = WH;
    c = grad_next_h;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, a, k, b, k, 0, c, n);

}

// grad_next_c = grad_next_c * f
// len:  N * H
static MKLNN_(LSTMFullStep_bprob_grad_next_c)(
  real * grad_next_c,
  real * ft,
  int len
)
{
   for(int i=0;i<len;i++)
   {
      grad_next_c[i] = grad_next_c[i] * ft[i];
   }
}


// x:        T, N, D
// WX size:  4, D, H
// WH size:  4, H, H
// gradOutpu:T, N, H
// h  size:  T, N, H
// c  size:  T, N, H
// grad_x:   T, N, D
// grad_b:      4, H
// grad_c0:     N, H
// grad_h0:     N, H
// gatesize: T,4N, H
//grad_buffer: 4N, H
void MKLNN_(LSTMFullStep_updateGradInput)(
  THTensor * x,
  THTensor * WX,
  THTensor * WH,
  THTensor * gradOutput,
  THTensor * h,
  THTensor * c,
  THTensor * h0,
  THTensor * c0,
  THTensor * gates,
  THTensor * grad_X,
  THTensor * grad_b,
  THTensor * grad_c0,
  THTensor * grad_h0,
  THTensor * grad_WX,
  THTensor * grad_WH,
  THTensor * grad_buffer1,
  THTensor * grad_buffer2
)
{
   int T = x->size[0];
   int N = x->size[1];
   int D = x->size[2];
   int H = grad_h0->size[1];
   printf("RealLSTMFullStep_updateGradInput start\n");
   printf("T = %d, N = %d, D = %d, H = %d \n", T, N, D, H);


   gates = THTensor_(newContiguous)(gates);

   real * x_base = THTensor_(data)(x);
   real * xt = NULL;

   real * gate_base = THTensor_(data)(gates);
   real * it = NULL;
   real * ft = NULL;
   real * ot = NULL;
   real * gt = NULL;
   real * grad_a = THTensor_(data)(grad_buffer1);
   real * grad_a2= THTensor_(data)(grad_buffer2);
   real * grad_it = NULL;
   real * grad_ft = NULL;
   real * grad_ot = NULL;
   real * grad_gt = NULL;

   real * Wx = THTensor_(data)(WX);
   real * Wh = THTensor_(data)(WH);

   real * grad_x_base = THTensor_(data)(grad_X);
   real * grad_x = NULL;
   real * grad_Wx = THTensor_(data)(grad_WX);
   real * grad_Wh = THTensor_(data)(grad_WH);
   real scale = 1.0;

   real * grad_h = THTensor_(data)(gradOutput);
   real * grad_ht = NULL;
   int t = 0;

   real * grad_next_h = THTensor_(data)(grad_h0);
   real * grad_next_c = THTensor_(data)(grad_c0);
   //memcpy(grad_next_h,  THTensor_(data)(gradOutput) + (T-1)*N*H , N * H * sizeof(real));
   real * next_c = NULL;
   real * prev_c = NULL;
   real * prev_h = NULL;
   for(t =T-1; t >= 0 ; t--)
   {
      xt = x_base + t * N * D;
      it = gate_base + t * 4 * N * H;
      ft = gate_base + t * 4 * N * H +     N * H;
      ot = gate_base + t * 4 * N * H + 2 * N * H;
      gt = gate_base + t * 4 * N * H + 3 * N * H;
      grad_it = grad_a;
      grad_ft = grad_a + N * H;
      grad_ot = grad_a + 2 * N * H;
      grad_gt = grad_a + 3 * N * H;
 
      grad_x = grad_x_base + + t * N * D;
      next_c = THTensor_(data)(c) + t * N * H;
      if(t==0)
      {
         prev_c = THTensor_(data)(c0) ;
         prev_h = THTensor_(data)(h0) ;
      }
      else
      {
         prev_c = THTensor_(data)(c) + (t-1) * N * H;
         prev_h = THTensor_(data)(h) + (t-1) * N * H;
      }
      grad_ht = grad_h + t*N*H;
      MKLNN_(LSTMFullStep_Add)(grad_next_h,grad_ht,N*H);
/*
      if(t == T-2)
      {
         MKLNN_(LSTMFullStep_PrintSum)(next_c,N*H,"next_c sum =");
         MKLNN_(LSTMFullStep_PrintSum)(grad_next_h,N*H,"grad_next_h sum =");
         MKLNN_(LSTMFullStep_PrintSum)(ot,N*H,"ot sum =");
         //MKLNN_(LSTMFullStep_PrintSum)(grad_next_c,N*H,"grad_next_c 1");
      }
*/
      //calc grad_next_c, grad_gate_o
      MKLNN_(LSTMFullStep_bprob_gateCO)(next_c,grad_next_c,ot,grad_ot,grad_next_h, N*H);
      MKLNN_(LSTMFullStep_bprob_gateIFG)(it,ft,gt,prev_c,grad_it,grad_ft,grad_gt,grad_next_c, N*H);
      MKLNN_(LSTMFullStep_bprob_activation)(it,ft,ot,gt,grad_it,grad_ft,grad_ot,grad_gt,grad_next_c,N*H);
      MKLNN_(LSTMFullStep_bprob_transpose)(grad_a,grad_a2,N,H);
      MKLNN_(LSTMFullStep_bprob_linear)(xt,prev_h,Wx,Wh,grad_a2,grad_x,grad_Wx,grad_Wh,grad_next_h,scale,T,N,D,H);
      MKLNN_(LSTMFullStep_bprob_grad_next_c)(grad_next_c,ft,N*H);
      //if(t == T-2)
      {
         printf("------------------------------------- t = %d\n",t);
         MKLNN_(LSTMFullStep_PrintSum)(grad_ot,N*H,"grad_ot");
         MKLNN_(LSTMFullStep_PrintSum)(grad_gt,N*H,"grad_gt");
         MKLNN_(LSTMFullStep_PrintSum)(grad_it,N*H,"grad_it");
         MKLNN_(LSTMFullStep_PrintSum)(grad_ft,N*H,"grad_ft");
         MKLNN_(LSTMFullStep_PrintSum)(grad_a, N*4*H,"grad_a");
         MKLNN_(LSTMFullStep_PrintSum)(Wx, D*4*H,"Wx");
         MKLNN_(LSTMFullStep_PrintSum)(grad_x, N*D,"grad_x");
         MKLNN_(LSTMFullStep_PrintSum)(grad_Wx,D*4*H,"grad_Wx");
         MKLNN_(LSTMFullStep_PrintSum)(grad_Wh,H*4*H,"grad_Wh");
         MKLNN_(LSTMFullStep_PrintSum)(grad_next_h,N*H,"grad_next_h");
         MKLNN_(LSTMFullStep_PrintSum)(grad_next_c,N*H,"grad_next_c");

      }


   }




}


#endif

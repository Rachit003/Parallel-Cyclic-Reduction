#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void init_f(double *f, double delta)
{
    int i = threadIdx.x;
    f[i]=sin(5*i*delta);
}

__global__ void init_abcd(double *a, double *b, double *c, double *d, double *f, double delta){
    int i = threadIdx.x;
    if(i != 0){
        a[i] = 1.0;
        b[i] = 4.0;
        c[i] = 1.0;
        d[i] = 3*(f[i+1]-f[i-1])/delta;
    }
}

__global__ void equate(double *a, double *b, double *c, double *d, double *a1, double *b1, double *c1, double *d1){
    int i = threadIdx.x;
    a1[i]=a[i];
    b1[i]=b[i];
    c1[i]=c[i];
    d1[i]=d[i];
}

__global__ void cyclic_reduction_loop(double *a, double *b, double *c, double *d, double *a1, double *b1, double *c1, double *d1, int n, int k){
    int i = (pow(2,k)-1) + threadIdx.x*pow(2,k);
    double alpha_i,beta_i;
    int up,dwn;
    up = i-pow(2,k-1);
    dwn= i+pow(2,k-1);
    alpha_i= (-1)*a[i]/b[up];
    beta_i= (-1)*c[i]/b[dwn];
    //printf("%d , %d , %d\n",i,up,dwn);
    if(i-pow(2,k)<0)
    {
        a1[i]=0;
    }
    else
    {
        a1[i]=a[up]*alpha_i;
    }
    if(dwn>=n)
    {
        b1[i]=b[i]+c[up]*alpha_i;
    }
    else{
        b1[i]=b[i]+a[dwn]*beta_i+c[up]*alpha_i;
    }

    if(dwn>=n)
    {
        c1[i]=0;
    }
    else
    {
        c1[i]=c[dwn]*beta_i;
    }
    if(dwn>=n)
    {
        d1[i]=d[i]+alpha_i*d[up];
    }
    else{
        d1[i]=d[i]+alpha_i*d[up]+beta_i*d[dwn];
    }
}

__global__ void cyclic_reduction_loop(double *a, double *b, double *c, double *d, int n, int k){
    int i = (pow(2,k)-1) + threadIdx.x*pow(2,k+1);
    double sub_alpha, sub_beta;
    int up,dwn;
    up=i-pow(2,k);
    dwn=i+pow(2,k);
    if(up<0)
    {
        sub_beta=-c[i]/b[dwn];
        d1[i]=d[i]+d[dwn]*sub_beta;
    }
    else
    {
        if(dwn>n-1)
        {
            sub_alpha=-a[i]/b[up];
            d1[i]=d[i]+d[up]*sub_alpha;
        }
        else
        {
            sub_beta=-c[i]/b[dwn];
            sub_alpha=-a[i]/b[up];
            d1[i]=d[i]+d[up]*sub_alpha+d[dwn]*sub_beta;
        }
    }
}

__global__ void backprop(double *b, double *d){
    int i = threadIdx.x;
    d[i]/=b[i];
}

int main()
{
    clock_t start=clock();
    int n=1001;
    cudaMalloc((void**)&a, n*sizeof(double));
    cudaMalloc((void**)&b, n*sizeof(double));
    cudaMalloc((void**)&c, n*sizeof(double));
    cudaMalloc((void**)&d, n*sizeof(double));
    cudaMalloc((void**)&f, n*sizeof(double));
    double delta=0.003;

    init_f<<< 1,n >>>(f, delta);
    init_abcd<<< 1,n-1 >>>(a,b,c,d,f,delta);

    a[0]=0;
    b[0]=1;
    c[0]=2;
    c[n-1]=0;
    a[n-1]=2;
    b[n-1]=1;
    d[n-1]=(2.5*f[n-1]-2*f[n-2]-0.5*f[n-3])/delta;
    d[0]=(-2.5*f[0]+2*f[1]+0.5*f[2])/delta;
    
    cudaMalloc((void**)&a1, n*sizeof(double));
    cudaMalloc((void**)&b1, n*sizeof(double));
    cudaMalloc((void**)&c1, n*sizeof(double));
    cudaMalloc((void**)&d1, n*sizeof(double));
    int logn;
    logn = log(n)/log(2);
    //// cyclic reduction 
    for(int k=1;k<=logn;k++)
    {
        equate<<< 1,n >>>(a,b,c,d,a1,b1,c1,d1);
        int n_threads = (n - (pow(2,k)-1))/pow(2,k);
        cyclic_reduction_loop<<< 1,n_threads >>>(a,b,c,d,a1,b1,c1,d1,n,k);
        equate<<< 1,n >>>(a1,b1,c1,d1,a,b,c,d);

    }
    for(int k=logn-1; k>=0;k--)
    {
        equate<<< 1,n >>>(a,b,c,d,a1,b1,c1,d1);
        int n_threads = (n - (pow(2,k)-1))/pow(2,k+1);
        cyclic_reduction_loop<<< 1,n_threads >>>(a,b,c,d,n,k);
        equate<<< 1,n >>>(a1,b1,c1,d1,a,b,c,d);
    }
    backprop<<< 1,n >>>(b,d);

    double d_host[n];
    cudaMemcpy(d_host, d, n*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++)
    {
        printf("%lf ,", d_host[i]);
    }
    clock_t end=clock();
    printf("Time: %ld\n",end-start);
    return 0;
}

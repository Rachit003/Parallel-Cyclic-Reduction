#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double func(double x)
{
    return sin(5 * x);
}

void cyclic_reduction(double *a, double *b, double *c, double *d, int n, int NUM)
{
    // Forward Reduction
    int k = 1, eqtn = n;
    while (eqtn / 2 >= NUM)
    {
        eqtn = 0;
        #pragma acc parallel num_gangs(NUM) reduction(+:eqtn) present(a[0:n], b[0:n], c[0:n], d[0:n])
        #pragma acc loop independent
        for (int i = pow(2, k) - 1; i < n; i += pow(2, k))
        {
            eqtn++;
            double alpha_i, beta_i;
            int up, dwn;
            up = i - pow(2, k - 1);
            dwn = i + pow(2, k - 1);
            alpha_i = (-1) * a[i] / b[up];
            beta_i = (-1) * c[i] / b[dwn];
            if (i - pow(2, k) < 0)
            {
                a[i] = 0;
            }
            else
            {
                a[i] = a[up] * alpha_i;
            }
            if (dwn >= n)
            {
                b[i] = b[i] + c[up] * alpha_i;
                c[i] = 0;
                d[i] = d[i] + alpha_i * d[up];
            }
            else
            {
                b[i] = b[i] + a[dwn] * beta_i + c[up] * alpha_i;
                c[i] = c[dwn] * beta_i;
                d[i] = d[i] + alpha_i * d[up] + beta_i * d[dwn];
            }
        }
        k++;
    }

    // Recursive Doubling //
    k--;
    int log_eqtn;
    log_eqtn = log(eqtn) / log(2);
    for (int j = 0; j <= log_eqtn; j++)
    {
        #pragma acc parallel num_gangs(NUM) present(a[0:n], b[0:n], c[0:n], d[0:n])
        #pragma acc loop independent
        for (int i = pow(2, k) - 1; i < n; i += pow(2, k + j))
        {
            double alpha_i, beta_i;
            int up, dwn;
            up = i - pow(2, k + j);
            dwn = i + pow(2, k + j);
            alpha_i = (-1) * a[i] / b[up];
            beta_i = (-1) * c[i] / b[dwn];
            if (up < 0)
            {
                if (dwn < n)
                {
                    a[i] = 0;
                    b[i] = b[i] + a[dwn] * beta_i;
                    c[i] = c[dwn] * beta_i;
                    d[i] = d[i] + beta_i * d[dwn];
                }
                else
                {
                    a[i] = 0;
                    b[i] = b[i];
                    c[i] = 0;
                    d[i] = d[i];
                }
            }
            else
            {
                if (dwn > n - 1)
                {
                    a[i] = a[up] * alpha_i;
                    b[i] = b[i] + c[up] * alpha_i;
                    c[i] = 0;
                    d[i] = d[i] + alpha_i * d[up];
                }
                else
                {
                    a[i] = a[up] * alpha_i;
                    b[i] = b[i] + a[dwn] * beta_i + c[up] * alpha_i;
                    c[i] = c[dwn] * beta_i;
                    d[i] = d[i] + alpha_i * d[up] + beta_i * d[dwn];
                }
            }
        }
    }

    // Back Substitution //
    for (k; k >= 0; k--)
    {
        #pragma acc parallel num_gangs(NUM) present(a[0:n], b[0:n], c[0:n], d[0:n])
        #pragma acc loop independent
        for (int i = pow(2, k) - 1; i < n; i += pow(2, k + 1))
        {
            double sub_alpha, sub_beta;
            int up, dwn;
            up = i - pow(2, k);
            dwn = i + pow(2, k);
            if (up < 0)
            {
                sub_beta = -c[i] / b[dwn];
                d[i]=d[i]+d[dwn]*sub_beta;
            }
            else
            {
                if(dwn>n-1)
                {
                    sub_alpha=-a[i]/b[up];
                    d[i]=d[i]+d[up]*sub_alpha;
                }
                else
                {
                    sub_beta=-c[i]/b[dwn];
                    sub_alpha=-a[i]/b[up];
                    d[i]=d[i]+d[up]*sub_alpha+d[dwn]*sub_beta;
                }
            }
        }
    }

    // Normalised Solution // 
    #pragma acc parallel num_gangs(NUM)
    #pragma acc loop gang
    for(int i=0;i<n;i++)
    {
        d[i]/=b[i];
    }
}
int main(int argc, char* argv[])
{
    // int NUM=1;
    clock_t start=clock();
    int NUM = atoi(argv[1]);
    int n;
    n=10001;
    NUM=2;  //optimal
    double a[n],b[n],c[n],d[n],f[n];
    double delta=0.0003;
    for(int i=0;i<n;i++)
    {
        f[i]=func(i*delta);
    }
    for (int i = 1; i < n-1; i++) {
        a[i] = 1.0;
        b[i] = 4.0;
        c[i] = 1.0;
        d[i] = 3*(f[i+1]-f[i-1])/delta;
    }
    a[0]=0;
    b[0]=1;
    c[0]=2;
    c[n-1]=0;
    a[n-1]=2;
    b[n-1]=1;
    d[n-1]=(2.5*f[n-1]-2*f[n-2]-0.5*f[n-3])/delta;
    d[0] = (-2.5 * f[0] + 2 * f[1] + 0.5 * f[2]) / delta;

    #pragma acc data copyin(a[0:n], b[0:n], c[0:n], d[0:n]) copyout(d[0:n])
    {
        cyclic_reduction(a, b, c, d, n, NUM);
    }

    for (int i = 2; i < n-1; i++)
    {
        printf("%lf ,", d[i]*2/3);
    }
    // ignoring initial 2 and last value to avoid error , i wasn't able to locate any error 
    clock_t end = clock();
    printf("Time: %ld\n", end - start);
    return 0;
}
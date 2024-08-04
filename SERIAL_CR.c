#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double func(double x)
{
    return sin(5*x);
}

void cyclic_reduction(double a[],double b[],double c[],double d[], int n)
{
    double a1[n],b1[n],c1[n],d1[n];
    int logn;
    logn = log(n)/log(2);
    //// cyclic reduction 
    for(int k=1;k<=logn;k++)
    {
        for(int i=0;i<n;i++)
        {
            a1[i]=a[i];
            b1[i]=b[i];
            c1[i]=c[i];
            d1[i]=d[i];
        }
        for(int i=pow(2,k)-1;i<n;i+=pow(2,k))
        {
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
        for(int i=0;i<n;i++)
        {
            a[i]=a1[i];
            b[i]=b1[i];
            c[i]=c1[i];
            d[i]=d1[i];
        }
    }
    for(int k=logn-1; k>=0;k--)
    {
        for(int i=0;i<n;i++)
        {
            a1[i]=a[i];
            b1[i]=b[i];
            c1[i]=c[i];
            d1[i]=d[i];
        }
        for(int i=pow(2,k)-1;i<n;i+=pow(2,k+1))
        {
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
        for(int i=0;i<n;i++)
        {
            a[i]=a1[i];
            b[i]=b1[i];
            c[i]=c1[i];
            d[i]=d1[i];
        }
    }
    for(int i=0;i<n;i++)
    {
        d[i]/=b[i];
    }
}
int main()
{
    clock_t start=clock();
    int n;
    n=1001;
    double a[n],b[n],c[n],d[n],f[n];
    double delta=0.003;
    //printf("%lf\n",delta);
    for(int i=0;i<n;i++)
    {
        f[i]=func(i*delta);
        //printf("%lf , ",f[i]);
    }
    //printf("\n");
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
    d[0]=(-2.5*f[0]+2*f[1]+0.5*f[2])/delta;
    cyclic_reduction(a,b,c,d,n);
    for(int i=0;i<n;i++)
    {
        printf("%lf ,", d[i]);
    }
    clock_t end=clock();
    printf("Time: %ld\n",end-start);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>

#include "dotprod-krnl.h"

int main(int argc, char* argv[])
{   
    int i = 0;
    float * a = (float *)malloc(sizeof(float)*VS); 
    float * b = (float *)malloc(sizeof(float)*VS); 
    float * c = (float *)malloc(sizeof(float)*1); 
    float * c_acc = (float *)malloc(sizeof(float)*1);

    // init
    for(i = 0; i < VS; i++) {
        a[i] = float(i) / VS;
        b[i] = float(i * i) / VS;
    }    

    printf("Starting Dot Product Example \n");

    // execute original reference code
    c[0] = 0.f;
    for (int i = 0; i < VS; ++i) { 
        c[0] += a[i] * b[i];
    }

    // execute kernel code 
    dot_product(a, b, c_acc);

    // test results
    bool err = false;
    if ((c[0] - c_acc[0]) / c[0] > 1e-4 || (c[0] - c_acc[0]) / c[0] < -1e-4) {
        err = true;
    }

    free(a);
    free(b);
    free(c);
    free(c_acc);

    if (err) {
        printf("Test failed %d\n",err);
        return 1;
    }
    else {
        printf("Test passed\n");
        return 0;
    }
}


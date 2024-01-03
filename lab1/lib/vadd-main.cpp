#include <stdio.h>
#include <stdlib.h>

#include "vadd-krnl.h"

int main(int argc, char* argv[])
{   
    int i = 0;
    float * a = (float *)malloc(sizeof(float)*VS); 
    float * b = (float *)malloc(sizeof(float)*VS); 
    float * c = (float *)malloc(sizeof(float)*VS); 
    float * c_acc = (float *)malloc(sizeof(float)*VS);

    // init
    for(i = 0; i < VS; i++) {
        a[i] = float(i) / VS;
        b[i] = float(i * i) / VS;
    }    

    printf("Starting Dot Product Example \n");

    // execute original reference code
    for (int i = 0; i < VS; ++i) { 
        c[i] = a[i] * b[i];
    }

    // execute kernel code 
    vadd(a, b, c_acc);

    // test results
    bool err = false;
    for (int i = 0; i < VS; ++i) { 
        if ((c[i] - c_acc[i]) / c[i] > 1e-4 || (c[i] - c_acc[i]) / c[i] < -1e-4) {
            err = true;
            break;
        }
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


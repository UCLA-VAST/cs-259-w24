#include <ap_int.h>

// parameters
#define kNum            (256)
#define kKernel         (5)
#define kImSize         (224)
#define kInImSize       (228)
#define kOutImSize      (112)
#define max(X,Y) ((X)>(Y)?(X):(Y))
// software golden model
void CnnKernelSoftware(
    const float input[kNum][kInImSize][kInImSize],
    const float weight[kNum][kNum][kKernel][kKernel],
    const float bias[kNum],
    float output[kNum][kOutImSize][kOutImSize]
  ) {

  // Allocate memory on heap to avoid stack overflow.
  static float C[kNum][kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = bias[i];
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              C[i][h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
            }
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = max(0, C[i][h][w]);
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output[i][h][w] = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
      }
    }
  }
}

extern "C" { 
  // TODO: Modify the data types based on your kernel implementation.
  void CnnKernel(class ap_uint< 128 > merlin_input[3326976],float weight[1638400],const class ap_uint< 512 > bias[16],class ap_uint< 512 > merlin_output[200704]);
}

int main(){
  printf("Starting C-simulation...\n");
  // input data
  static float input[kNum][kInImSize][kInImSize];
  // weight data
  static float weight[kNum][kNum][kKernel][kKernel];
  // bias data
  static float bias[kNum];
  // golden output data
  static float outputSoftware[kNum][kOutImSize][kOutImSize];
  // hardware output data
  static float outputHardware[kNum][kOutImSize][kOutImSize];

  // initialize input data
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kInImSize; ++h) {
      for (int w = 0; w < kInImSize; ++w) {
        input[i][h][w] = (float)rand() / RAND_MAX;
      }
    }
  }

  // initialize weight data
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q) {
          weight[i][j][p][q] = (float)rand() / RAND_MAX;
        }
      }
    }
  }

  // initialize bias data
  for (int i = 0; i < kNum; ++i) {
    bias[i] = (float)rand() / RAND_MAX;
  }

  // call the software golden model
  CnnKernelSoftware(input, weight, bias, outputSoftware);
  printf("Software finished!\n");

  // call the hardware function
  // TODO: Modify the data types based on your kernel implementation.
  CnnKernel((ap_uint< 128 > *)input,(float *)weight,(ap_uint< 512 > *)bias,(ap_uint< 512 > *)outputHardware);
  printf("Hardware finished!\n");

  // compare the results of software and hardware
  int err = 0;
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        if(abs(outputSoftware[i][h][w] - outputHardware[i][h][w]) > 0.1){
          err++;
        }
      }
    }
  }

  if(err == 0){
    printf("Test passed!\n");
    return 0;
  }else{
    printf("Test failed with %d errors!\n", err);
    return 1;
  }
}

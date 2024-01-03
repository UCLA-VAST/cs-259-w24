#include "vadd-krnl.h"

#pragma ACCEL kernel
void vadd(const float a[VS], const float b[VS], float c[VS]) {
  #pragma ACCEL pipeline
  for (int i = 0; i < VS; ++i) {
    c[i] = a[i] * b[i];
  }
}
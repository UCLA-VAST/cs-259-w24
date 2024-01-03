#include "dotprod-krnl.h"

#pragma ACCEL kernel
void dot_product(const float a[VS], const float b[VS], float c[0]) {
  float prod = 0.f;

  #pragma ACCEL pipeline
  for (int i = 0; i < VS; ++i) {
    prod += a[i] * b[i];
  }
  c[0] = prod;
}
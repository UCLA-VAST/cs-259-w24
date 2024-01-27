#define __constant
#define __kernel
#define __global
#include "memcpy_128_1d.h"
#include "memcpy_512_1d.h"
#define SIZE_1 228
#include "memcpy_128_2d.h"
#undef SIZE_1
#define SIZE_1 112
#define SIZE_2 112
#include "memcpy_512_3d.h"
#undef SIZE_1
#undef SIZE_2
#include <string.h> 

#include "merlin_type_define.h"







// Original: #pragma ACCEL kernel
static int CnnKernel_dummy_pos;

static void merlin_memcpy_0(float dst[256][256][5][5],int dst_idx_0,int dst_idx_1,int dst_idx_2,int dst_idx_3,float src[1638400],int src_idx_0,unsigned int len,unsigned int max_len)
{
  
#pragma HLS inline off
  
#pragma HLS function_instantiate variable=dst_idx_0,dst_idx_1,dst_idx_2,dst_idx_3,src_idx_0
  unsigned long i;
  unsigned long total_offset1 = (((0 * 256 + dst_idx_0) * 256 + dst_idx_1) * 5 + dst_idx_2) * 5 + dst_idx_3;
  unsigned long total_offset2 = 0 * 1638400 + src_idx_0;
  merlinL0:
  for (i = 0; i < len / 4; ++i) {
    
#pragma HLS PIPELINE II=1
    
#pragma HLS LOOP_TRIPCOUNT max=1638400
    dst[(total_offset1 + i) / 5 / 5 / 256][(total_offset1 + i) / 5 / 5 % 256][(total_offset1 + i) / 5 % 5][(total_offset1 + i) % 5] = src[total_offset2 + i];
  }
}
extern "C" { 

void CnnKernel(class ap_uint< 128 > merlin_input[3326976],float weight[1638400],const class ap_uint< 512 > bias[16],class ap_uint< 512 > merlin_output[200704])
{
  
#pragma HLS INTERFACE m_axi port=bias offset=slave depth=16 bundle=merlin_gmem_CnnKernel_512_0
  
#pragma HLS INTERFACE m_axi port=merlin_input offset=slave depth=3326976 bundle=merlin_gmem_CnnKernel_128_0
  
#pragma HLS INTERFACE m_axi port=merlin_output offset=slave depth=200704 bundle=merlin_gmem_CnnKernel_512_0
  
#pragma HLS INTERFACE m_axi port=weight offset=slave depth=1638400 bundle=merlin_gmem_CnnKernel_32_0
  
#pragma HLS INTERFACE s_axilite port=bias bundle=control
  
#pragma HLS INTERFACE s_axilite port=merlin_input bundle=control
  
#pragma HLS INTERFACE s_axilite port=merlin_output bundle=control
  
#pragma HLS INTERFACE s_axilite port=weight bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=merlin_output
  
#pragma HLS DATA_PACK VARIABLE=bias
  
#pragma HLS DATA_PACK VARIABLE=merlin_input
  
#pragma ACCEL interface variable=merlin_output depth=256,112,112 max_depth=256,112,112
  
#pragma ACCEL interface variable=bias depth=256 max_depth=256
  
#pragma ACCEL interface variable=weight depth=256,256,5,5 max_depth=256,256,5,5
  
#pragma ACCEL interface variable=merlin_input depth=256,228,228 max_depth=256,228,228
  float bias_10_0_buf[256];
  
#pragma HLS array_partition variable=bias_10_0_buf cyclic factor=16 dim=1
  float weight_16_0_buf[256][256][5][5];
  
#pragma HLS array_partition variable=weight_16_0_buf complete dim=4
  
#pragma HLS array_partition variable=weight_16_0_buf complete dim=3
  float merlin_output_buf[256][112][112];
  
#pragma HLS array_partition variable=merlin_output_buf cyclic factor=16 dim=3
// Allocate memory on heap to avoid stack overflow.
  static float C[256][224][224];
  
#pragma HLS array_partition variable=C cyclic factor=2 dim=3
  
#pragma HLS array_partition variable=C cyclic factor=2 dim=2
{
    memcpy_wide_bus_read_float_512(&bias_10_0_buf[0],(class ap_uint< 512 > *)bias,(0 * 4),sizeof(float ) * ((unsigned long )256),256L);
  }
  merlinL15:
  for (int i = 0; i < 256; ++i) {
    merlinL14:
    for (int h = 0; h < 224; ++h) {
      merlinL13:
      for (int w = 0; w < 224; ++w) 
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=C array inter false
        
#pragma HLS pipeline
        C[i][h][w] = bias_10_0_buf[i];
      }
    }
  }
{
    merlin_memcpy_0(weight_16_0_buf,0,0,0,0,weight,0,sizeof(float ) * ((unsigned long )1638400L),6553600UL);
  }
// Convolution
  merlinL12:
// Convolution
  for (int i = 0; i < 256; ++i) {
    merlinL11:
    for (int j = 0; j < 256; ++j) {
      float merlin_input_16_0_buf[228][228];
      
#pragma HLS array_partition variable=merlin_input_16_0_buf cyclic factor=5 dim=2
      
#pragma HLS array_partition variable=merlin_input_16_0_buf cyclic factor=5 dim=1
{
        memcpy_wide_bus_read_float_2d_228_128(merlin_input_16_0_buf,(::size_t )0,(::size_t )0,(class ap_uint< 128 > *)merlin_input,(::size_t )(((long )j) * 51984L * ((long )4)),sizeof(float ) * ((unsigned long )51984L),(::size_t )51984L);
      }
      merlinL10:
      for (int h = 0; h < 224; ++h) {
        merlinL9:
        for (int w = 0; w < 224; ++w) 
// Original: #pragma ACCEL PIPELINE AUTO
{
          
#pragma HLS dependence variable=C array inter false
          
#pragma HLS pipeline
          merlinL8:
          for (int p = 0; p < 5; ++p) 
// Original: #pragma ACCEL PARALLEL COMPLETE
{
            
#pragma HLS unroll
            merlinL7:
            for (int q = 0; q < 5; ++q) 
// Original: #pragma ACCEL PARALLEL COMPLETE
{
              
#pragma HLS unroll
              C[i][h][w] += weight_16_0_buf[i][j][p][q] * merlin_input_16_0_buf[h + p][q + w];
            }
          }
        }
      }
    }
// Existing HLS partition: #pragma HLS array_partition variable=merlin_input_16_0_buf cyclic factor = 4 dim=2
  }
// ReLU
  merlinL6:
// ReLU
  for (int i = 0; i < 256; ++i) {
    merlinL5:
    for (int h = 0; h < 224; ++h) {
      merlinL4:
      for (int w = 0; w < 224; ++w) 
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=C array inter false
        
#pragma HLS pipeline
        if (((float )0) > C[i][h][w]) {
          C[i][h][w] = ((float )0);
        }
         else {
          C[i][h][w] = C[i][h][w];
        }
      }
    }
  }
// Max pooling
  merlinL3:
// Max pooling
  for (int i = 0; i < 256; ++i) {
    merlinL2:
    for (int h = 0; h < 112; ++h) {
      merlinL1:
      for (int w = 0; w < 112; ++w) 
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=merlin_output_buf array inter false
        
#pragma HLS pipeline
        char rose__temp = (char )true;
        float rose_temp;
        if (C[i][h * 2][w * 2] > C[i][h * 2 + 1][w * 2]) {
          rose_temp = C[i][h * 2][w * 2];
        }
         else {
          rose_temp = C[i][h * 2 + 1][w * 2];
        }
        float rose_temp_0;
        if (C[i][h * 2][w * 2 + 1] > C[i][h * 2 + 1][w * 2 + 1]) {
          rose_temp_0 = C[i][h * 2][w * 2 + 1];
        }
         else {
          rose_temp_0 = C[i][h * 2 + 1][w * 2 + 1];
        }
        rose__temp = ((char )((bool )(rose_temp > rose_temp_0)));
        if ((bool )rose__temp) {
          if (C[i][h * 2][w * 2] > C[i][h * 2 + 1][w * 2]) {
            merlin_output_buf[i][h][w] = C[i][h * 2][w * 2];
          }
           else {
            merlin_output_buf[i][h][w] = C[i][h * 2 + 1][w * 2];
          }
        }
         else {
          if (C[i][h * 2][w * 2 + 1] > C[i][h * 2 + 1][w * 2 + 1]) {
            merlin_output_buf[i][h][w] = C[i][h * 2][w * 2 + 1];
          }
           else {
            merlin_output_buf[i][h][w] = C[i][h * 2 + 1][w * 2 + 1];
          }
        }
      }
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=merlin_output_buf cyclic factor = 16 dim=3
  memcpy_wide_bus_write_float_3d_112_112_512((class ap_uint< 512 > *)merlin_output,merlin_output_buf,0,0,0,(4 * 0),sizeof(float ) * ((unsigned long )3211264L),3211264L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=bias_10_0_buf cyclic factor = 16 dim=1

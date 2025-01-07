#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) 
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) 
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif

#include <immintrin.h>

extern "C" DLL_PUBLIC void im2col(float* __restrict  input_img, 
                                  short width, 
                                  short height, 
                                  short rowBlock, 
                                  short colBlock, 
                                  float* __restrict outTensor) {
    int yB = height - rowBlock + 1;      // Output rows
    int xB = width - colBlock + 1;      // Output columns
    int k_size = rowBlock * colBlock;   // Flattened block size

    __m256 source_vector;              // AVX register
    int numBlocks = (colBlock + 7) / 8; 
    if(numBlocks==1){
    // ======================================================
    // Case 1: numBlocks == 1  ==> colBlock <= 8
    // Only 1 load/store of 8 floats per row-block
    // ======================================================
      for (int i = 0; i < yB; i++) {     
              for (int j = 0; j < xB; j++) { 
                  for (int k = 0; k < rowBlock; k++) { 
                      int input_offset = (i + k) * width + j;  
                      int output_offset = (i * xB * k_size) 
                                        + (j * k_size) 
                                        + (k * colBlock); 
                      source_vector = _mm256_loadu_ps(&input_img[input_offset]);
                      _mm256_storeu_ps(&outTensor[output_offset], source_vector);
                  }
              }
    }
    }else{
    // ======================================================
    // Case 2: numBlocks > 2  ==> colBlock > 8
    // We do numBlocks loads/stores in a small inner loop.
    // Each block = 8 floats.
    // ======================================================
      for (int i = 0; i < yB; i++) {     
          for (int j = 0; j < xB; j++) { 
              for (int k = 0; k < rowBlock; k++) { 
                  int input_offset = (i + k) * width + j;  
                  int output_offset = (i * xB * k_size) 
                                    + (j * k_size) 
                                    + (k * colBlock); 

                  for (int l = 0; l < numBlocks; l++) { // Process each 8-float block
                      source_vector = _mm256_loadu_ps(&input_img[input_offset + l * 8]);
                      _mm256_storeu_ps(&outTensor[output_offset + l * 8], source_vector);
                  }
              }
          }
      }
    }
}
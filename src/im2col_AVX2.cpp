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
      for (int i = 0; i < yB; i++) {     // Slide over height
              for (int j = 0; j < xB; j++) { // Slide over width
                  for (int k = 0; k < rowBlock; k++) { // Row blocks
                      int input_offset = (i + k) * width + j;  // Input offset
                      int output_offset = (i * xB * k_size) + (j * k_size) + (k * colBlock); // Output offset
                      source_vector = _mm256_loadu_ps(&input_img[input_offset]);
                      _mm256_storeu_ps(&outTensor[output_offset], source_vector);
                  }
              }
    }
    }else{
      for (int i = 0; i < yB; i++) {     // Slide over height
          for (int j = 0; j < xB; j++) { // Slide over width
              for (int k = 0; k < rowBlock; k++) { // Row blocks
                  int input_offset = (i + k) * width + j;  // Input offset
                  int output_offset = (i * xB * k_size) + (j * k_size) + (k * colBlock); // Output offset

                  for (int l = 0; l < numBlocks; l++) { // Process each 8-float block
                      // Load 8 floats from input
                      source_vector = _mm256_loadu_ps(&input_img[input_offset + l * 8]);
                      
                      // Store 8 floats to output
                      _mm256_storeu_ps(&outTensor[output_offset + l * 8], source_vector);
                  }
              }
          }
      }
    }
}
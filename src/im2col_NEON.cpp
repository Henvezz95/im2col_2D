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

#include <arm_neon.h>
#include <cstdint>

extern "C" void im2col(
    float* __restrict input_img,
    short width,
    short height,
    short rowBlock,
    short colBlock,
    float* __restrict outTensor)
{
    int yB = height - rowBlock + 1;    // Output rows
    int xB = width - colBlock + 1;     // Output columns
    int k_size = rowBlock * colBlock;  // Flattened block size

    // Each block is 4 floats (128 bits)
    int numBlocks = (colBlock + 3) / 4;

    if (numBlocks == 1) {
        // ======================================================
        // Case 1: numBlocks == 1  ==> colBlock <= 4
        // Only 1 load/store of 4 floats per row-block
        // ======================================================
        for (int i = 0; i < yB; i++) {
            for (int j = 0; j < xB; j++) {
                for (int k = 0; k < rowBlock; k++) {
                    int input_offset =
                        (i + k) * width + j;  
                    int output_offset =
                        (i * xB * k_size) + (j * k_size) + (k * colBlock);

                    float32x4_t v0 = vld1q_f32(&input_img[input_offset]);
                    vst1q_f32(&outTensor[output_offset], v0);
                }
            }
        }

    } else if (numBlocks == 2) {
        // ======================================================
        // Case 2: numBlocks == 2  ==> 5 <= colBlock <= 8
        // Two loads/stores of 4 floats per row-block
        // ======================================================
        for (int i = 0; i < yB; i++) {
            for (int j = 0; j < xB; j++) {
                for (int k = 0; k < rowBlock; k++) {
                    int input_offset =
                        (i + k) * width + j;
                    int output_offset =
                        (i * xB * k_size) + (j * k_size) + (k * colBlock);
                    
                    float32x4_t v0 = vld1q_f32(&input_img[input_offset]);
                    vst1q_f32(&outTensor[output_offset], v0);
                    float32x4_t v1 = vld1q_f32(&input_img[input_offset + 4]);
                    vst1q_f32(&outTensor[output_offset + 4], v1);
                }
            }
        }

    } else {
        // ======================================================
        // Case 3: numBlocks > 2  ==> colBlock > 8
        // We do numBlocks loads/stores in a small inner loop.
        // Each block = 4 floats.
        // ======================================================
        for (int i = 0; i < yB; i++) {
            for (int j = 0; j < xB; j++) {
                for (int k = 0; k < rowBlock; k++) {
                    int input_offset  = (i + k) * width + j;
                    int output_offset = (i * xB * k_size)
                                      + (j * k_size)
                                      + (k * colBlock);

                    for (int l = 0; l < numBlocks; l++) {
                        float32x4_t v0 = vld1q_f32(&input_img[input_offset + l * 4]);
                        vst1q_f32(&outTensor[output_offset + l * 4], v0);
                    }
                }
            }
        }
    }
}
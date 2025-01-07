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

#include <cstdint>

// A simple reference implementation of im2col.
// No SIMD intrinsics or special optimizations.
extern "C" void im2col(const float* input_img,
                       short width,
                       short height,
                       short rowBlock,
                       short colBlock,
                       float* outTensor)
{
    // Number of "sliding window" positions in the vertical (height) direction
    int yB = height - rowBlock + 1;
    // Number of "sliding window" positions in the horizontal (width) direction
    int xB = width - colBlock + 1;
    // Total number of elements in one "block" (rowBlock x colBlock)
    int k_size = rowBlock * colBlock;

    // Slide over the height
    for (int i = 0; i < yB; i++) {
        // Slide over the width
        for (int j = 0; j < xB; j++) {
            // For each row in the block
            for (int rb = 0; rb < rowBlock; rb++) {
                // For each column in the block
                for (int cb = 0; cb < colBlock; cb++) {

                    // Compute where to read from in the input
                    int input_offset  = (i + rb) * width + (j + cb);

                    // Compute where to write in the output
                    //   - Each (i,j) patch has k_size elements
                    //   - (rb * colBlock + cb) indexes within that patch
                    int output_offset = (i * xB * k_size)
                                      + (j * k_size)
                                      + (rb * colBlock + cb);

                    // Copy one float from the input image into the output tensor
                    outTensor[output_offset] = input_img[input_offset];
                }
            }
        }
    }
}

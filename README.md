=====================================================
README - OpenCL Manual Mirrored Repeat Sampler Test
=====================================================

Overview
--------
This project demonstrates how to implement "mirrored repeat" image sampling 
manually in OpenCL. The OpenCL standard provides CL_ADDRESS_MIRRORED_REPEAT, 
but some drivers have broken implementations. This code provides a reliable 
CPU and GPU implementation to verify results and work around those issues.

The program:
 - Creates a small input image (4x5).
 - Expands it into a larger output image (12x12).
 - Applies an offset (uint2) to shift the sampling coordinates. This ensures
   that mirrored repeat effects are visible on all output borders.
 - Runs both a CPU reference calculation and an OpenCL GPU kernel that 
   manually implements mirrored repeat.
 - Compares the results between CPU and GPU and prints the matrices.

Files
-----
main.cpp         : Host code that sets up input data, runs the CPU reference, 
                   compiles the OpenCL kernel, executes it, and compares results.
sampler_test.cl  : OpenCL kernel source implementing manual mirrored repeat 
                   with support for an uint2 offset parameter.

How It Works
------------
1. Input data is a simple 1D array (filled with 0..N-1).
2. The CPU reference function maps output coordinates into the input image 
   using a mirrored repeat function. This mirrors the image outwards in 
   both x and y directions, like a tiling effect where every other tile is flipped.
3. The OpenCL kernel performs the same mapping on the GPU.
4. The results from CPU and GPU are compared to ensure correctness.

Offset Parameter
----------------
The "offset" is a uint2 (two unsigned integers) passed to the kernel. 
It shifts the input coordinates relative to the output image. 
This allows the mirrored repeat boundaries to appear on all edges 
of the larger output image rather than just on two sides.

For example:
 - Without offset, mirroring might only be visible on right/bottom edges.
 - With offset (e.g., {5, 2}), you can see mirrored boundaries on all sides.

Build Instructions
------------------
1. Ensure you have an OpenCL SDK installed (e.g., Intel, AMD, or NVIDIA).
2. Compile the host code with a C++ compiler and link against OpenCL.
   Example (Linux, GCC):
       g++ main.cpp -lOpenCL -o mirrored_test
   Example (Windows, MSVC):
       cl main.cpp OpenCL.lib
3. Ensure "sampler_test.cl" is in the same directory as the executable.

Run
---
Execute the program:
    ./mirrored_test

It will:
 - Run the CPU reference calculation.
 - Run the OpenCL kernel.
 - Compare the two results.
 - Print both matrices for inspection.

Expected Output
---------------
The console should show:
   --- Results Comparison ---
   CPU vs. Manual Mirrored Kernel: PASS

Then, it will print both the CPU and Kernel results in matrix form.
If they match, the test passes.

Notes
-----
 - This example uses 2D images (image2d_t in OpenCL).
 - The channel format is CL_R with CL_FLOAT for simplicity.
 - Error checking is performed using the CL_CHECK macro.
 - If you see mismatches, check the OpenCL driver version or offset handling.

License
-------
You may use and modify this code freely for testing, educational, or debugging purposes.

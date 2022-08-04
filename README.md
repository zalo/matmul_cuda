# matmul_cuda
 A simple learning example for CUDA.
 
 Take in a `.bin` file specifying a series of `int32 rows, int32 cols, int64_t* matrixContents` matrices and multiply them together as quickly as possible.  Save the resultant matrix as `output.bin` in the same format.

On Windows, this application compiles with:
```
nvcc -o matmul_cuda_v2 matmul_cuda.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" -O 3 -arch=all --extra-device-vectorization
```

On Linux, this application compiles with:
```
nvcc -o matmul_cuda_v2 matmul_cuda.cu -O 3 -arch=all --extra-device-vectorization
```

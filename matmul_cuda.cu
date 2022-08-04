#include <stdio.h>
#include <stdint.h>

// CUDA Kernel for multiplying matrices
__global__
void matmul(int workingShapeX, int workingShapeY, int currentShapeY, int newShapeY, 
            int64_t *mat1, int64_t *mat2, int64_t *out) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(i < workingShapeX && j < workingShapeY){
    int64_t sum = 0;
    for (int k = 0; k < currentShapeY; k++) {
      sum += mat1[(i *  currentShapeY) + k] * 
             mat2[(k *      newShapeY) + j];
    }
    out[(i * workingShapeY) + j] = sum;
  }
}

int main(int argc, char *argv[]) {
  FILE* input = fopen(argv[1], "rb");
  if (input != NULL) {
    int *currentShape = (int *)malloc(8);
    int *workingShape = (int *)malloc(8);
    int *    newShape = (int *)malloc(8);
    int64_t *currentMatrix = NULL, *workingMatrix = NULL;

    // Read the number of Rows and Columns in this Matrix
    while ((fread(newShape, sizeof(char), 8, input)) > 0) {
      // Allocate this matrix
      int64_t* newMatrix = (int64_t*)malloc(newShape[0] * newShape[1] * sizeof(int64_t));
      // Read the full contents of the matrix
      int matrixBytes = fread(newMatrix, sizeof(int64_t), newShape[0] * newShape[1], input);

      bool readyToMultiply = currentMatrix != NULL;
      if(readyToMultiply) {
        // Allocate the Working Matrix
        workingShape[0]   = currentShape[0];
        workingShape[1]   =     newShape[1];
        int workingLength = workingShape[0] * workingShape[1] * sizeof(int64_t);
        int currentLength = currentShape[0] * currentShape[1] * sizeof(int64_t);
        int     newLength =     newShape[0] *     newShape[1] * sizeof(int64_t);
        workingMatrix = (int64_t*)malloc(workingLength);

        // Multiply Current and New Matrices together
        if (workingLength > 10000) { // Choose GPU or CPU based on Matrix Size
          // Allocate Matrices on GPU
          int64_t *d_workingMatrix; cudaMalloc(&d_workingMatrix, workingLength);
          int64_t *d_currentMatrix; cudaMalloc(&d_currentMatrix, currentLength);
          int64_t *d_newMatrix    ; cudaMalloc(&    d_newMatrix,     newLength);

          // Copy Matrices to GPU
          cudaMemcpy(d_currentMatrix, currentMatrix, currentLength, cudaMemcpyHostToDevice);
          cudaMemcpy(    d_newMatrix,     newMatrix,     newLength, cudaMemcpyHostToDevice);

          // Multiply Matrices on GPU
          dim3 threadsPerBlock(32, 32); // 1024 is typically the max allowable threads per block
          dim3 numBlocks((workingShape[0] / threadsPerBlock.x) + 1,
                         (workingShape[1] / threadsPerBlock.y) + 1);
          matmul<<<numBlocks, threadsPerBlock>>>( workingShape[0], workingShape[1], currentShape[1], newShape[1],
                                                  d_currentMatrix, d_newMatrix, d_workingMatrix);

          // Copy Solution back from GPU
          cudaMemcpy(workingMatrix, d_workingMatrix, workingLength, cudaMemcpyDeviceToHost);

          // Free Memory on GPU
          cudaFree(d_workingMatrix);
          cudaFree(d_currentMatrix);
          cudaFree(    d_newMatrix);
        } else {
          // Naive CPU Fallback for Small Matrices
          for (int i = 0; i < currentShape[0]; i++) {       // i is the row in this matrix
            for (int j = 0; j <     newShape[1]; j++) {     // j is the column in the other matrix
              workingMatrix[(i * workingShape[1]) + j] = 0; // malloc doesn't initialize to zeros on Linux...
              for (int k = 0; k < currentShape[1]; k++) {   // k is the column in this matrix
                workingMatrix[(i * workingShape[1]) + j] +=
                currentMatrix[(i * currentShape[1]) + k] *
                    newMatrix[(k *     newShape[1]) + j];
              }
            }
          }
        }
        free(newMatrix);
      }

      free(currentMatrix);
      currentMatrix   = readyToMultiply ? workingMatrix   : newMatrix;
      currentShape[0] = readyToMultiply ? workingShape[0] : newShape[0];
      currentShape[1] = readyToMultiply ? workingShape[1] : newShape[1];
    }

    fclose (input); // Reached the end of the input file; close the input here

    // Write the output matrix here
    FILE* output = fopen("output.bin", "wb");
    fwrite(currentShape, 4, 2, output);
    fwrite(currentMatrix, 8, currentShape[0] * currentShape[1], output);
    fclose(output);

    free(currentMatrix);
    free(currentShape);
    free(workingShape);
    free(    newShape);
  }
}

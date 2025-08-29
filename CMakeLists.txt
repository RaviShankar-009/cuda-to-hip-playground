cmake_minimum_required(VERSION 3.18)
project(cuda_to_hip_playground LANGUAGES C CXX CUDA)

add_executable(vectorAdd_cuda cuda_src/vectorAdd.cu)
set_target_properties(vectorAdd_cuda PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  CUDA_ARCHITECTURES "60;70;75;80;86;89"
)

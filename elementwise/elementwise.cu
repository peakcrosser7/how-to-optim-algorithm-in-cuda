#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <chrono>

using namespace std;

#define N 32 * 1024 * 1024
// elementwise implementation copyed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

/// @brief 获取kernel启动的gridSize大小
/// @param n element-wise处理的数据总数
/// @param[out] num_blocks 设置的线程块数
inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count; // SM个数
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;  // SM中线程最大数
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, // 按数据个数取整划分线程块(数据量比较小)
                                                   sm_count * tpm / kBlockSize * kNumWaves)); // 按GPU线程处理量划分(数据量比较大)
  return cudaSuccess;
}

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

/// @brief T类型合并pack_size个的打包结构
template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

/// @brief 向量化内存访问最大字节数
constexpr int kMaxPackBytes = 128 / 8;
/// @brief 最大合并读取个数
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

/// @brief T类型数据可以合并读取的个数
template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

/// @brief 多个类型可合并读取个数的最小值
template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

template<typename T>
class HasApply2 {
  typedef char one;
  struct two {
    char x[2];
  };

  /// 如果类型C有Apply2函数则匹配成功
  template<typename C>
  static one test(decltype(&C::Apply2));
  template<typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
  // 对向量化合并读取的元素两个两个的处理
#pragma unroll
  for (int j = 0; j < pack_size; j += 2) { functor.Apply2(ret.elem + j, (in.elem + j)...); }
  return ret;
}

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
  // 对向量化合并读取的元素逐一处理
#pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in.elem[j])...); }
  return ret;
}

/// @brief element-wise执行通用内核函数
/// @tparam pack_size 向量化合并的元素个数
/// @param factory 仿函数容器
/// @param n_pack 向量化合并的元素分组数
/// @param[out] pack_r 合并的元素输出
/// @param ...pack_in 合并的元素输入
/// @param n_tail 尾部元素个数
/// @param[out] tail_r 尾部元素输出
/// @param ...tail_in 尾部元素输入
template<int pack_size, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory(); // 仿函数
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  // 处理向量化合并的元素
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
  }
  // 处理尾部元素
  if (global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : tpl(functor) {}
  __device__ FunctorT operator()() const { return tpl; }

 private:
  FunctorT tpl;
};

template<size_t pack_size>
bool IsAlignedForPack() {
  return true;
}

/// @brief 判断多个指针是否都内存pack_size字节对齐
template<size_t pack_size, typename T, typename... Args>
bool IsAlignedForPack(const T* ptr, const Args*... others) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0
         && IsAlignedForPack<pack_size, Args...>(others...);
}

template<size_t pack_size, typename FactoryT, typename R, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R* r, const IN*... in) {
  // 向量化合并后的分组数
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  // 合并后剩余不够合并的个数
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0>>>(
      factory, n_pack, reinterpret_cast<Packed<R, pack_size>*>(r),
      (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
      (in + tail_offset)...);
  return cudaPeekAtLastError();
}

template<typename FactoryT, typename R, typename... IN>
struct GenericLauncher {
  /// @brief 启动内核函数
  /// @param factory 仿函数
  /// @param n 元素个数
  /// @param r 输出值
  /// @param ...in 输入值
  /// @return 是否成功
  static cudaError_t Launch(FactoryT factory, int64_t n, R* r, const IN*... in) {
    constexpr int max_pack_size = PackSize<R, IN...>();
    if (IsAlignedForPack<max_pack_size, R, IN...>(r, in...)) {  // 内存对齐可以向量化读取
      return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, n, r, in...);
    } else {    // 内存未对齐
      return LaunchKernel<1, FactoryT, R, IN...>(factory, n, r, in...);
    }
  }
};

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, n, r, a);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int64_t n, R* r, const A* a) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a);
}

template<typename FactoryT, typename R, typename A, typename B>
inline cudaError_t BinaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b) {
  return GenericLauncher<FactoryT, R, A, B>::Launch(factory, n, r, a, b);
}

template<typename FunctorT, typename R, typename A, typename B>
inline cudaError_t Binary(FunctorT functor, int64_t n, R* r, const A* a, const B* b) {
  return BinaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b);
}

template<typename FactoryT, typename R, typename A, typename B, typename C>
inline cudaError_t TernaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                      const C* c) {
  return GenericLauncher<FactoryT, R, A, B, C>::Launch(factory, n, r, a, b, c);
}

template<typename FunctorT, typename R, typename A, typename B, typename C>
inline cudaError_t Ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c) {
  return TernaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, c);
}

template<typename T>
struct MultiplyFunctor {
  __device__ T operator()(T x, T y) const {
    return x*y;
  }
};

template<>
struct MultiplyFunctor<half> {
  __device__ half operator()(half x, half y) const {
    return x*y;
  }
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* z, const half* x, const half* y) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const half2 y2 = *(reinterpret_cast<const half2*>(y));
    *reinterpret_cast<half2*>(z) = __hmul2(x2, y2);
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};

template<typename T>
__global__ void mul(T *x, T *y, T* z){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  z[idx] = x[idx] * y[idx];
}

template<>
__global__ void mul(half *x, half *y, half* z){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  z[idx] = x[idx] * y[idx];
}

/// manual element-wise kernel with Packed
template <int pack_size>
__global__ void mul_coalesced(half *x, half *y, half* z, int64_t n){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t n_pack = n / pack_size;
  int64_t pack_off = n_pack * pack_size;

  auto pack_x = (reinterpret_cast<Packed<half, pack_size>*>(x));
  auto pack_y = (reinterpret_cast<Packed<half, pack_size>*>(y));
  auto pack_z = (reinterpret_cast<Packed<half, pack_size>*>(z));
  for (int i = idx; i < n_pack; i += gridDim.x * blockDim.x) {
    auto half_x = pack_x[i];
    auto half_y = pack_y[i];
    Packed<half, pack_size> half_z;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      half_z.elem[j] = half_x.elem[j] * half_y.elem[j];
    }
    pack_z[i] = half_z;
  }
  for (int i = pack_off + idx; i < n; i += gridDim.x * blockDim.x) {
    z[i] = x[i] * y[i];
  }
}




int main(){
    std::chrono::high_resolution_clock::time_point t1, t2;

    half *x_host = (half*)malloc(N*sizeof(half));
    half *x_device;
    cudaMalloc((void **)&x_device, N*sizeof(half));
    for (int i = 0; i < N; i++) x_host[i] = 2.0;
    cudaMemcpy(x_device, x_host, N*sizeof(half), cudaMemcpyHostToDevice);

    half *y_host = (half*)malloc(N*sizeof(half));
    half *y_device;
    cudaMalloc((void **)&y_device, N*sizeof(half));
    for (int i = 0; i < N; i++) y_host[i] = 2.0;
    cudaMemcpy(y_device, y_host, N*sizeof(half), cudaMemcpyHostToDevice);

    half *output_host = (half*)malloc(N * sizeof(half));
    half *output_device;
    cudaMalloc((void **)&output_device, N * sizeof(half));

    // naive elementwise
    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);
    t1 = std::chrono::high_resolution_clock::now();
    mul<half><<<grid, block>>>(x_device, y_device, output_device);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cout << "naive   elementwise cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns\n";
    cudaMemcpy(output_host, output_device, N * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        cout << __half2float(output_host[i]) << " ";
    }
    cout << endl;
    // coalesced mul
    constexpr auto pack_size = PackSize<half>();
    GetNumBlocks(N / pack_size, &block_num);
    t1 = std::chrono::high_resolution_clock::now();
    mul_coalesced<pack_size><<<block_num, kBlockSize>>>(x_device, y_device, output_device, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cout << "coalesc elementwise cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns\n";
    cudaMemcpy(output_host, output_device, N * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        cout << __half2float(output_host[i]) << " ";
    }
    cout << endl;
    // elementwise template
    t1 = std::chrono::high_resolution_clock::now();
    Binary(MultiplyFunctor<half>(), N, output_device, x_device, y_device);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cout << "oneflow elementwise cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns\n";
    cudaMemcpy(output_host, output_device, N * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        cout << __half2float(output_host[i]) << " ";
    }
    cout << endl;
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
    return 0;
}

// float dtype
// int main(){
//     float *x_host = (float*)malloc(N*sizeof(float));
//     float *x_device;
//     cudaMalloc((void **)&x_device, N*sizeof(float));
//     for (int i = 0; i < N; i++) x_host[i] = 2.0;
//     cudaMemcpy(x_device, x_host, N*sizeof(float), cudaMemcpyHostToDevice);

//     float *y_host = (float*)malloc(N*sizeof(float));
//     float *y_device;
//     cudaMalloc((void **)&y_device, N*sizeof(float));
//     for (int i = 0; i < N; i++) y_host[i] = 2.0;
//     cudaMemcpy(y_device, y_host, N*sizeof(float), cudaMemcpyHostToDevice);

//     float *output_host = (float*)malloc(N * sizeof(float));
//     float *output_device;
//     cudaMalloc((void **)&output_device, N * sizeof(float));

//     // naive elementwise
//     int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
//     dim3 grid(block_num, 1);
//     dim3 block(kBlockSize, 1);
//     mul<float><<<grid, block>>>(x_device, y_device, output_device);
//     cudaMemcpy(output_host, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);

//     // elementwise template
//     Binary(MultiplyFunctor<float>(), N, output_device, x_device, y_device);
//     cudaMemcpy(output_host, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);
//     free(x_host);
//     free(y_host);
//     free(output_host);
//     cudaFree(x_device);
//     cudaFree(y_device);
//     cudaFree(output_device);
//     return 0;
// }

extern "C" __global__ void noop() {}

extern "C" __global__ void addf(const float* a, const float* b, float* c,
                                unsigned long long n) {
    unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void chase(const unsigned int* next,
                                 unsigned long long start,
                                 unsigned long long steps,
                                 unsigned int* out) {
    unsigned long long p = start;
    for (unsigned long long i = 0; i < steps; i++) {
        p = next[p];
    }
    *out = (unsigned int)p;
}

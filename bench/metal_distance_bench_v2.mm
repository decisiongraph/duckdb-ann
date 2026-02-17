/// Metal distance benchmark v2: pre-allocated buffers to isolate dispatch overhead.
/// Tests whether buffer allocation or command submission is the bottleneck.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static float cpu_l2(const float *a, const float *b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static void cpu_batch_l2(const float *query, const float *cands, int n, int dim, float *out) {
    for (int i = 0; i < n; i++)
        out[i] = cpu_l2(query, cands + i * dim, dim);
}

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; i++)
        data[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char **argv) {
    const char *metallib = "build/release/faiss_metal.metallib";
    if (argc > 1)
        metallib = argv[1];
    srand(42);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal\n");
            return 1;
        }
        printf("Metal: %s\n\n", [[device name] UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSString *path = [NSString stringWithUTF8String:metallib];
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!lib) {
            fprintf(stderr, "No metallib\n");
            return 1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"diskann_batch_l2"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];

        struct Config {
            int n;
            int dim;
            int iters;
        };
        Config configs[] = {
            {64, 128, 1000}, {64, 256, 1000},  {64, 768, 500},  {128, 768, 500}, {128, 1536, 200},
            {256, 768, 200}, {256, 1536, 100}, {512, 1536, 50}, {1024, 768, 50},
        };

        for (auto &cfg : configs) {
            int n = cfg.n, dim = cfg.dim, iters = cfg.iters;
            printf("--- n=%d, dim=%d (n*dim=%d) ---\n", n, dim, n * dim);

            std::vector<float> query(dim);
            std::vector<float> cands(n * dim);
            std::vector<float> cpu_out(n), gpu_out(n);
            fill_random(query.data(), dim);
            fill_random(cands.data(), n * dim);

            size_t q_sz = dim * sizeof(float);
            size_t c_sz = (size_t)n * dim * sizeof(float);
            size_t o_sz = n * sizeof(float);
            struct {
                uint32_t n;
                uint32_t dim;
            } params = {(uint32_t)n, (uint32_t)dim};

            // Pre-allocate buffers (reused across iterations)
            id<MTLBuffer> q_buf = [device newBufferWithLength:q_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> c_buf = [device newBufferWithLength:c_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> o_buf = [device newBufferWithLength:o_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> p_buf = [device newBufferWithBytes:&params
                                                      length:sizeof(params)
                                                     options:MTLResourceStorageModeShared];

            // Warm up
            cpu_batch_l2(query.data(), cands.data(), n, dim, cpu_out.data());
            memcpy([q_buf contents], query.data(), q_sz);
            memcpy([c_buf contents], cands.data(), c_sz);
            {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:q_buf offset:0 atIndex:0];
                [enc setBuffer:c_buf offset:0 atIndex:1];
                [enc setBuffer:o_buf offset:0 atIndex:2];
                [enc setBuffer:p_buf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // CPU benchmark
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++)
                cpu_batch_l2(query.data(), cands.data(), n, dim, cpu_out.data());
            auto t1 = Clock::now();
            double cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // GPU benchmark (pre-allocated buffers, only copy + dispatch)
            t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                memcpy([q_buf contents], query.data(), q_sz);
                memcpy([c_buf contents], cands.data(), c_sz);
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:q_buf offset:0 atIndex:0];
                [enc setBuffer:c_buf offset:0 atIndex:1];
                [enc setBuffer:o_buf offset:0 atIndex:2];
                [enc setBuffer:p_buf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                memcpy(gpu_out.data(), [o_buf contents], o_sz);
            }
            t1 = Clock::now();
            double gpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // GPU no-copy: data already in shared buffers, just dispatch
            t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:q_buf offset:0 atIndex:0];
                [enc setBuffer:c_buf offset:0 atIndex:1];
                [enc setBuffer:o_buf offset:0 atIndex:2];
                [enc setBuffer:p_buf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
            t1 = Clock::now();
            double gpu_nc_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // Verify
            memcpy(gpu_out.data(), [o_buf contents], o_sz);
            float max_err = 0;
            for (int i = 0; i < n; i++) {
                float err = fabs(cpu_out[i] - gpu_out[i]);
                if (err > max_err)
                    max_err = err;
            }

            printf("  CPU:          %8.1f us\n", cpu_us);
            printf("  GPU (copy):   %8.1f us  (%.2fx)\n", gpu_us, cpu_us / gpu_us);
            printf("  GPU (no-copy):%8.1f us  (%.2fx)\n", gpu_nc_us, cpu_us / gpu_nc_us);
            printf("  Max err: %.2e\n\n", max_err);
        }
    }
    return 0;
}

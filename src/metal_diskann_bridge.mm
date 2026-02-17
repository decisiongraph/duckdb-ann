#ifdef FAISS_METAL_ENABLED

#include "include/metal_diskann_bridge.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>

namespace {

struct DiskannMetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> l2_pipeline = nil;
    id<MTLComputePipelineState> ip_pipeline = nil;
    bool initialized = false;

    bool init() {
        if (initialized)
            return true;

        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device)
                return false;

            queue = [device newCommandQueue];
            if (!queue)
                return false;

            // Load metallib (same path used by faiss-metal)
            NSString *path = @FAISS_METAL_METALLIB_PATH;
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:path];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            if (!library)
                return false;

            id<MTLFunction> l2_fn = [library newFunctionWithName:@"diskann_batch_l2"];
            id<MTLFunction> ip_fn = [library newFunctionWithName:@"diskann_batch_ip"];
            if (!l2_fn || !ip_fn)
                return false;

            l2_pipeline = [device newComputePipelineStateWithFunction:l2_fn error:&error];
            if (!l2_pipeline)
                return false;

            ip_pipeline = [device newComputePipelineStateWithFunction:ip_fn error:&error];
            if (!ip_pipeline)
                return false;

            initialized = true;
        }
        return true;
    }

    static DiskannMetalState &instance() {
        static DiskannMetalState state;
        return state;
    }
};

} // anonymous namespace

extern "C" int diskann_metal_available(void) {
    auto &state = DiskannMetalState::instance();
    if (!state.initialized) {
        state.init();
    }
    return state.initialized ? 1 : 0;
}

extern "C" int diskann_metal_batch_distances(const float *query, const float *candidates, int n, int dim, int metric,
                                             float *out_distances) {
    if (n <= 0 || dim <= 0 || !query || !candidates || !out_distances) {
        return -1;
    }

    auto &state = DiskannMetalState::instance();
    if (!state.initialized && !state.init()) {
        return -1;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = (metric == 0) ? state.l2_pipeline : state.ip_pipeline;

        size_t query_size = (size_t)dim * sizeof(float);
        size_t candidates_size = (size_t)n * dim * sizeof(float);
        size_t distances_size = (size_t)n * sizeof(float);

        // Create shared buffers (zero-copy on Apple Silicon unified memory)
        id<MTLBuffer> query_buf = [state.device newBufferWithBytes:query
                                                            length:query_size
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> candidates_buf = [state.device newBufferWithBytes:candidates
                                                                 length:candidates_size
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> distances_buf = [state.device newBufferWithLength:distances_size
                                                                options:MTLResourceStorageModeShared];

        // Params buffer (matches DiskannDistParams layout)
        struct {
            uint32_t n;
            uint32_t dim;
        } params = {(uint32_t)n, (uint32_t)dim};
        id<MTLBuffer> params_buf = [state.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        if (!query_buf || !candidates_buf || !distances_buf || !params_buf) {
            return -1;
        }

        id<MTLCommandBuffer> cmd_buf = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:query_buf offset:0 atIndex:0];
        [encoder setBuffer:candidates_buf offset:0 atIndex:1];
        [encoder setBuffer:distances_buf offset:0 atIndex:2];
        [encoder setBuffer:params_buf offset:0 atIndex:3];

        // One threadgroup per candidate, 32 threads per threadgroup (1 simdgroup)
        MTLSize grid_size = MTLSizeMake(n, 1, 1);
        MTLSize group_size = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];

        [encoder endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (cmd_buf.status == MTLCommandBufferStatusError) {
            return -1;
        }

        memcpy(out_distances, [distances_buf contents], distances_size);
    }

    return 0;
}

#endif // FAISS_METAL_ENABLED

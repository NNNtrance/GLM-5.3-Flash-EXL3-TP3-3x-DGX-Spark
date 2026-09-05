// Grid barrier vs kernel boundary, GB10 (sm_121).
// N trivial elementwise phases over one buffer, run as (a) N separate launches
// and (b) one cooperative kernel with grid.sync() between phases; each arm both
// inside a CUDA graph (stream capture) and outside.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static const int THREADS = 256;
static const int LAYERS  = 42;   // "42 layers"
static const int PHASES  = 3;    // three phases per layer
static const int REPS    = 42;   // 42 repetitions per measurement

__global__ void phase_kernel(float* __restrict__ buf, int n, float c)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        buf[i] = fmaf(buf[i], 1.000001f, c);
}

__global__ void coop_kernel(float* __restrict__ buf, int n, int phases)
{
    cg::grid_group grid = cg::this_grid();
    const int stride = grid.size();
    const int t0 = grid.thread_rank();
    for (int p = 0; p < phases; ++p)
    {
        const float c = (float) p;
        for (int i = t0; i < n; i += stride)
            buf[i] = fmaf(buf[i], 1.000001f, c);
        if (p < phases - 1) grid.sync();
    }
}

static void launch_coop(float* buf, int n, int grid, cudaStream_t s)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int phases = PHASES;
    CK(cudaLaunchKernelEx(&cfg, coop_kernel, buf, n, phases));
}

// one "step" = LAYERS layers, each of PHASES phases
static void step_separate(float* buf, int n, int grid, cudaStream_t s)
{
    for (int l = 0; l < LAYERS; ++l)
        for (int p = 0; p < PHASES; ++p)
            phase_kernel<<<grid, THREADS, 0, s>>>(buf, n, (float) p);
}
static void step_coop(float* buf, int n, int grid, cudaStream_t s)
{
    for (int l = 0; l < LAYERS; ++l) launch_coop(buf, n, grid, s);
}

template <class F>
static double median_ms(F f, cudaStream_t s)
{
    for (int i = 0; i < 5; ++i) f();          // warmup / discard first rounds
    CK(cudaStreamSynchronize(s));
    std::vector<double> ts;
    cudaEvent_t b, e; CK(cudaEventCreate(&b)); CK(cudaEventCreate(&e));
    for (int r = 0; r < REPS; ++r)
    {
        CK(cudaEventRecord(b, s));
        f();
        CK(cudaEventRecord(e, s));
        CK(cudaEventSynchronize(e));
        float ms; CK(cudaEventElapsedTime(&ms, b, e));
        ts.push_back(ms);
    }
    CK(cudaEventDestroy(b)); CK(cudaEventDestroy(e));
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

int main()
{
    int sms = 0, dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    sms = prop.multiProcessorCount;
    int coopSupported = 0;
    CK(cudaDeviceGetAttribute(&coopSupported, cudaDevAttrCooperativeLaunch, dev));

    int occ_coop = 0, occ_phase = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_coop,  (void*) coop_kernel,  THREADS, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_phase, (void*) phase_kernel, THREADS, 0));
    const int coop_grid = occ_coop * sms;

    printf("# device            : %s\n", prop.name);
    printf("# SMs               : %d\n", sms);
    printf("# cooperativeLaunch : %d\n", coopSupported);
    printf("# threads/block     : %d\n", THREADS);
    printf("# occupancy coop    : %d blocks/SM  -> max cooperative grid = %d blocks\n", occ_coop, coop_grid);
    printf("# occupancy phase   : %d blocks/SM  -> %d blocks\n", occ_phase, occ_phase * sms);
    printf("# layers=%d phases=%d reps=%d (median)\n", LAYERS, PHASES, REPS);
    printf("# both arms use the same persistent grid (%d blocks)\n\n", coop_grid);

    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int sizes[] = {16384, 262144, 4194304};

    printf("%10s %12s %12s %12s %12s %12s %12s\n",
           "N", "sep_ms", "coop_ms", "d_us/bnd", "sepG_ms", "coopG_ms", "dG_us/bnd");

    for (int si = 0; si < 3; ++si)
    {
        const int n = sizes[si];
        float* buf = nullptr;
        CK(cudaMalloc(&buf, (size_t) n * sizeof(float)));
        CK(cudaMemset(buf, 0, (size_t) n * sizeof(float)));

        double sep  = median_ms([&]{ step_separate(buf, n, coop_grid, s); }, s);
        double coop = median_ms([&]{ step_coop(buf, n, coop_grid, s); }, s);

        // --- CUDA graph arms (stream capture) ---
        double sepG = -1.0, coopG = -1.0;
        cudaGraph_t g1 = nullptr, g2 = nullptr;
        cudaGraphExec_t x1 = nullptr, x2 = nullptr;

        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        step_separate(buf, n, coop_grid, s);
        CK(cudaStreamEndCapture(s, &g1));
        CK(cudaGraphInstantiate(&x1, g1, nullptr, nullptr, 0));
        sepG = median_ms([&]{ CK(cudaGraphLaunch(x1, s)); }, s);

        cudaError_t cerr = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        if (cerr == cudaSuccess)
        {
            step_coop(buf, n, coop_grid, s);
            cerr = cudaStreamEndCapture(s, &g2);
            if (cerr == cudaSuccess && g2)
            {
                cerr = cudaGraphInstantiate(&x2, g2, nullptr, nullptr, 0);
                if (cerr == cudaSuccess)
                    coopG = median_ms([&]{ CK(cudaGraphLaunch(x2, s)); }, s);
                else printf("# coop graph instantiate failed: %s\n", cudaGetErrorString(cerr));
            }
            else printf("# coop graph capture failed: %s\n", cudaGetErrorString(cerr));
        }
        else printf("# coop graph begin-capture failed: %s\n", cudaGetErrorString(cerr));
        cudaGetLastError();

        // per-phase-boundary delta: LAYERS*PHASES boundaries per step
        const double bnd = (double) (LAYERS * PHASES);
        double d  = (coop  - sep)  * 1000.0 / bnd;
        double dG = (coopG > 0 && sepG > 0) ? (coopG - sepG) * 1000.0 / bnd : 0.0;

        printf("%10d %12.4f %12.4f %+12.3f %12.4f %12.4f %+12.3f\n",
               n, sep, coop, d, sepG, coopG, dG);

        if (x1) cudaGraphExecDestroy(x1);
        if (g1) cudaGraphDestroy(g1);
        if (x2) cudaGraphExecDestroy(x2);
        if (g2) cudaGraphDestroy(g2);
        CK(cudaFree(buf));
    }
    printf("\n# d_us/bnd = (coop - separate) us per phase boundary, %d boundaries/step\n", LAYERS * PHASES);
    printf("# positive = cooperative grid.sync costs MORE than a kernel boundary\n");
    return 0;
}

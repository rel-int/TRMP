// The forward segment kernel: see common.cuh for the layout and the phases.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"
static_assert(HIDP + PCELL_ROWS <= RA_ROWS, "cell-major per_cell fits under the per_cell rows");

template <typename T, int MODE>
__global__ void __launch_bounds__(NTHREADS, 1) segment_kernel(Params<T> p) {
    PHASE_BEGIN();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr bool ONCHIP = sizeof(T) == 4;
    T* base = ONCHIP ? reinterpret_cast<T*>(smem_raw)
                     : p.scratch + (size_t)blockIdx.x * (ROWS * MS + STAT_FLOATS);
    T* state = base;                       // [SDP][MS]
    T* ca = state + SDP * MS;              // [CAK][MS]: clue rows [0, DIMP), answer rows [DIMP, DIMP + YDP)
    T* answer = ca + DIMP * MS;
    T* ra = ca + CAK * MS;                 // region A [RA_ROWS][MS]
    T* rb = ra + RA_ROWS * MS;             // region B [RB_ROWS][MS]
    T* stat = rb + RB_ROWS * MS;           // [2][MS]
    const Offsets o = p.off;
    // the resident weights: in shared memory after the activations (float), or global
    // (double).  One packed buffer per layer, ``o.total`` apart; with KLAYERS > 1 the
    // slot holds the layer about to fire and is restaged after every application.
    const T* w = ONCHIP ? stat + STAT_FLOATS : p.weights;
    const auto layer_w = [&](int l) { return p.weights + (size_t)l * o.total; };
    if (ONCHIP) {
        // finite padding everywhere the fragment loads may reach
        for (int i = threadIdx.x; i < ROWS * MS + STAT_FLOATS; i += NTHREADS) base[i] = T(0);
        copy_words<T>(const_cast<T*>(w), p.weights, o.resident);
    } else {
        for (int i = threadIdx.x; i < ROWS * MS + STAT_FLOATS; i += NTHREADS) base[i] = T(0);
    }
    __syncthreads();
    Ports<T> ports;
    load_ports<T>(ports, p.graph);
    T gc[GC_ITEMS][3];
    // where the per-cycle maps are staged: the refresh maps above the refresh gates in
    // region A, the emit map in region B
    T* wr_s = ONCHIP ? ra + RUP(N3Y, 16) * MS : const_cast<T*>(p.weights + o.wrih);
    T* we_s = ONCHIP ? rb : const_cast<T*>(layer_w(KLAYERS - 1) + o.we);   // the last layer emits

    for (int b = blockIdx.x; b < p.B; b += gridDim.x) {
        T* rec = p.saves ? p.saves + (size_t)b * board_record(p.cycles, p.rounds * KLAYERS) : nullptr;
        T* crec = rec ? rec + (size_t)p.cycles * p.rounds * KLAYERS * ROUND_REC : nullptr;
        load_T<T, KSD, SDP>(state, p.state_in + (size_t)b * NC * KSD);
        load_T<T, KDIM, DIMP>(ca, p.clue_in + (size_t)b * NC * KDIM);
        load_T<T, KYD, YDP>(answer, p.answer_in + (size_t)b * NC * KYD);
        load_T<T, KDIM, DIMP>(rb, p.units_in + (size_t)b * NC * KDIM);   // the entry beliefs
        // Region B on entry: the beliefs (rows [0, DIMP)), then the staged entry port
        // map, then the staged clue/answer map; region A takes gconst.
        T* wp_s = rb + DIMP * MS;
        T* wca_s = wp_s + WP_ROWS * MS;
        if (ONCHIP) {
            copy_words<T>(wp_s, p.weights + o.wp, o.wca - o.wp);
            copy_words<T>(wca_s, p.weights + o.wca, o.bca - o.wca);
        } else {
            wp_s = const_cast<T*>(p.weights + o.wp);
            wca_s = const_cast<T*>(p.weights + o.wca);
        }
        __syncthreads();
        gemm<T, MODE, CAK, WS(CAK), NGF, Tile<N3S>::NT>(ra, ca, wca_s, p.weights + o.bca, N3S);
        __syncthreads();
        take_gc<T, KSD, GC_ITEMS>(gc, ra);
        __syncthreads();
        PHASE(0);
        bool entry = true;
        for (int cyc = 0; cyc < p.cycles; ++cyc) {
            // the layer applications of the cycle: ``rounds`` rounds of KLAYERS layers, in order
            for (int a = 0; a < p.rounds * KLAYERS; ++a) {
                const int l = a % KLAYERS;
                const T* wg = layer_w(l);          // layer l's buffer, for the staged blocks
                if (!ONCHIP) w = wg;               // (double: the resident group too)
                T* rk = rec ? rec + (size_t)(cyc * p.rounds * KLAYERS + a) * ROUND_REC : nullptr;
                if constexpr (KLAYERS > 1) {
                    // gconst is per layer and the registers hold one layer's: layer l's,
                    // from its clue|answer map staged into region B (free between
                    // applications; the entry application has it from above)
                    if (!entry) {
                        T* wca_l = ONCHIP ? rb : const_cast<T*>(wg + o.wca);
                        if (ONCHIP) copy_words<T>(wca_l, wg + o.wca, o.bca - o.wca);
                        __syncthreads();
                        gemm<T, MODE, CAK, WS(CAK), NGF, Tile<N3S>::NT>(ra, ca, wca_l, wg + o.bca, N3S);
                        __syncthreads();
                        take_gc<T, KSD, GC_ITEMS>(gc, ra);
                        __syncthreads();
                    }
                }
                if (rk) store_rows<T>(rk + SAVE_STATE, state, KSD);
                // G1a: shared | per_cell from the state
                gemm<T, MODE, SDP, WSS, NGF, Tile<N1A>::NT>(ra, state, w + o.w1a, w + o.b1a, N1A, 0, NWARPS, HIDP);
                if (entry) {
                    __syncthreads();
                    // the entry round reads the beliefs it was given (region B rows [0, DIMP))
                    gemm<T, MODE, DIMP, WS(KDIM), NGF, Tile<KHID>::NT>(ra + HIDP * MS, rb, wp_s, nullptr, KHID, 0, NWARPS, 0);
                    entry = false;
                }
                __syncthreads();
                PHASE(1);
                if (rk) { store_rows<T>(rk + SAVE_SHARED, ra, KHID); copy_block<T>(rk + SAVE_PCELL, ra + HIDP * MS, NC * PC_T); }
                pool<T>(rb, ra, ra + HIDP * MS, ports);
                __syncthreads();
                PHASE(2);
                if (rk) store_rows<T>(rk + SAVE_POOLED, rb, KHID);
                gemm<T, MODE, HIDP, WS(KHID), NGF, Tile<N3S>::NT>(ra, rb, w + o.w2p, nullptr, N3S);   // gi
                __syncthreads();
                PHASE(3);
                gemm<T, MODE, SDP, WSS, NGF, Tile<N3S>::NT>(rb, state, w + o.whh, w + o.bhh, N3S);  // gh
                __syncthreads();
                PHASE(4);
                if (rk) store_rows<T>(rk + SAVE_GHN, rb + 2 * KSD * MS, KSD);
                gru<T, KSD, GC_ITEMS>(state, ra, rb, gc, rk ? rk + SAVE_RZN : nullptr);
                __syncthreads();
                PHASE(5);
                layernorm<T, KSD>(state, w + o.lns_g, w + o.lns_b);
                if (ONCHIP && a == p.rounds * KLAYERS - 1)   // regions are free: stage the refresh maps
                    copy_words<T>(wr_s, p.weights + o.wrih, o.we - o.wrih);
                __syncthreads();
                if constexpr (ONCHIP && KLAYERS > 1) {
                    // the slot takes the next layer's resident group -- layer 0's after the
                    // last application, which is what the refresh reads its biases from
                    copy_words<T>(const_cast<T*>(w), layer_w((l + 1) % KLAYERS), o.resident);
                    __syncthreads();
                }
                PHASE(6);
            }
            // the refresh: two GEMMs in one phase, half the warps each; its blocks are
            // layer 0's (the slot, or layer 0's buffer)
            const T* w0 = ONCHIP ? w : p.weights;
            T* ck = crec ? crec + (size_t)cyc * CYC_REC : nullptr;
            if (ck) store_rows<T>(ck + CSAVE_ANS, answer, KYD);
            gemm<T, MODE, SDP, WSS, NGF, Tile<N3Y>::NT>(ra, state, wr_s, w0 + o.brih, N3Y, 0, NWARPS / 2);
            gemm<T, MODE, YDP, WSY, NGF, Tile<N3Y>::NT>(rb, answer, wr_s + (o.wrhh - o.wrih), w0 + o.brhh, N3Y, NWARPS / 2, NWARPS / 2);
            __syncthreads();
            PHASE(8);
            if (ck) store_rows<T>(ck + CSAVE_GHN, rb + 2 * KYD * MS, KYD);
            gru<T, KYD, (KYD * NC + NTHREADS - 1) / NTHREADS>(answer, ra, rb, nullptr, ck ? ck + CSAVE_RZN : nullptr);
            __syncthreads();
            layernorm<T, KYD>(answer, w0 + o.lny_g, w0 + o.lny_b);
            __syncthreads();
            PHASE(9);
            if (p.every) store_T<T, KYD>(p.every + ((size_t)cyc * p.B + b) * NC * KYD, answer);
            if constexpr (KLAYERS == 1) {
                // gconst for the next cycle: stage wca into region B (free now), GEMM into
                // region A (with more layers every application computes its own on entry)
                T* wca_b = ONCHIP ? rb : const_cast<T*>(p.weights + o.wca);
                if (ONCHIP) copy_words<T>(wca_b, p.weights + o.wca, o.bca - o.wca);
                __syncthreads();
                gemm<T, MODE, CAK, WS(CAK), NGF, Tile<N3S>::NT>(ra, ca, wca_b, p.weights + o.bca, N3S);
                __syncthreads();
                PHASE(10);
                take_gc<T, KSD, GC_ITEMS>(gc, ra);
            }
            if (ONCHIP && cyc == p.cycles - 1)     // region B is free: stage the emit map
                copy_words<T>(we_s, layer_w(KLAYERS - 1) + o.we, o.wp - o.we);
            __syncthreads();
            PHASE(11);
        }
        if (crec) {
            T* fin = crec + (size_t)p.cycles * CYC_REC;
            store_rows<T>(fin, state, KSD);
            store_rows<T>(fin + KSD * NCP, answer, KYD);
        }
        // emit: the last layer's map and bias (the slot holds layer 0's group by now)
        const T* be = (ONCHIP && KLAYERS > 1) ? layer_w(KLAYERS - 1) + o.be : w + o.be;
        gemm<T, MODE, SDP, WSS, 1, Tile<KDIM>::NT>(ra, state, we_s, be, KDIM);
        __syncthreads();
        store_T<T, KDIM>(p.units_out + (size_t)b * NC * KDIM, ra);
        store_T<T, KSD>(p.state_out + (size_t)b * NC * KSD, state);
        store_T<T, KYD>(p.answer_out + (size_t)b * NC * KYD, answer);
        __syncthreads();
        PHASE(12);
    }
}

static Offsets offsets_of(const std::vector<int64_t>& v) {
    Offsets o;
    int* f = reinterpret_cast<int*>(&o);
    TORCH_CHECK(v.size() == sizeof(Offsets) / sizeof(int), "offsets: wrong length");
    for (size_t i = 0; i < v.size(); ++i) f[i] = (int)v[i];
    return o;
}

template <typename T, int MODE>
static void launch(Params<T> p, int grid, cudaStream_t stream) {
    const size_t smem = sizeof(T) == 4
        ? (size_t)(ROWS * MS + STAT_FLOATS + p.off.resident) * sizeof(T) : 0;
    if (sizeof(T) == 4) {
        static bool configured = false;
        if (!configured) {
            TORCH_CHECK(smem <= 227 * 1024, "segment kernel: shared memory ", smem, " bytes");
            cudaFuncSetAttribute(segment_kernel<T, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            configured = true;
        }
    }
    segment_kernel<T, MODE><<<grid, NTHREADS, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
static Params<T> params_of(torch::Tensor& weights, const Offsets& off, const Wiring& g,
                           torch::Tensor& units, torch::Tensor& state, torch::Tensor& clue,
                           torch::Tensor& answer, torch::Tensor& units_out, torch::Tensor& state_out,
                           torch::Tensor& answer_out, torch::Tensor& every, bool want_every,
                           int64_t cycles, int64_t rounds) {
    Params<T> p;
    p.weights = weights.data_ptr<T>(); p.graph = graph_of<T>(g);
    p.units_in = units.data_ptr<T>(); p.state_in = state.data_ptr<T>();
    p.clue_in = clue.data_ptr<T>(); p.answer_in = answer.data_ptr<T>();
    p.units_out = units_out.data_ptr<T>(); p.state_out = state_out.data_ptr<T>();
    p.answer_out = answer_out.data_ptr<T>();
    p.every = want_every ? every.data_ptr<T>() : nullptr;
    p.scratch = nullptr; p.saves = nullptr; p.B = (int)units.size(0); p.cycles = (int)cycles;
    p.rounds = (int)rounds; p.off = off;
    return p;
}

std::vector<torch::Tensor> segment(torch::Tensor weights, std::vector<int64_t> offsets,
                                   torch::Tensor peers, torch::Tensor inverse, torch::Tensor masks,
                                   torch::Tensor rdeg, torch::Tensor units, torch::Tensor state,
                                   torch::Tensor clue, torch::Tensor answer, int64_t cycles,
                                   int64_t rounds, bool want_every, int64_t mode, bool want_saves) {
    const Wiring g = wiring_of(peers, inverse, masks, rdeg);
    TORCH_CHECK(units.is_cuda() && state.is_cuda() && clue.is_cuda() && answer.is_cuda());
    TORCH_CHECK(units.dim() == 3 && units.size(1) == NC && units.size(2) == KDIM, "units shape");
    TORCH_CHECK(state.size(1) == NC && state.size(2) == KSD, "state shape");
    TORCH_CHECK(clue.size(1) == NC && clue.size(2) == KDIM, "clue shape");
    TORCH_CHECK(answer.size(1) == NC && answer.size(2) == KYD, "answer shape");
    const c10::cuda::OptionalCUDAGuard guard(units.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int B = (int)units.size(0);
    units = units.contiguous(); state = state.contiguous(); clue = clue.contiguous();
    answer = answer.contiguous(); weights = weights.contiguous();
    auto units_out = torch::empty_like(units), state_out = torch::empty_like(state),
         answer_out = torch::empty_like(answer);
    auto every = want_every ? torch::empty({cycles, B, NC, KYD}, answer.options())
                            : torch::empty({0}, answer.options());
    auto saves = want_saves ? torch::empty({(int64_t)B * board_record((int)cycles, (int)rounds * KLAYERS)}, answer.options())
                            : torch::empty({0}, answer.options());
    // persistent blocks: one per SM (the shared-memory footprint allows one), each
    // looping over boards, so the resident weights are staged once per block
    static int n_sms = 0;
    if (n_sms == 0) cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, units.device().index());
    const int grid = B < 1 ? 1 : (B < n_sms ? B : n_sms);
    const Offsets off = offsets_of(offsets);
    // a net's layer count is a build parameter: a buffer of the wrong depth cannot run
    TORCH_CHECK(weights.numel() == (int64_t)KLAYERS * off.total, "weights: ", weights.numel(),
                " elements, the build has ", KLAYERS, " layer(s) of ", off.total);
    if (B > 0 && units.dtype() == torch::kFloat32) {
        auto p = params_of<float>(weights, off, g, units, state, clue, answer, units_out,
                                  state_out, answer_out, every, want_every, cycles, rounds);
        p.saves = want_saves ? saves.data_ptr<float>() : nullptr;
        MODE_DISPATCH(mode, launch<float, MODE_>(p, grid, stream));
    } else if (B > 0) {
        TORCH_CHECK(units.dtype() == torch::kFloat64, "float32 or float64");
        auto scratch = torch::empty({(int64_t)grid * (ROWS * MS + STAT_FLOATS)}, units.options());
        auto p = params_of<double>(weights, off, g, units, state, clue, answer, units_out,
                                   state_out, answer_out, every, want_every, cycles, rounds);
        p.scratch = scratch.data_ptr<double>();
        p.saves = want_saves ? saves.data_ptr<double>() : nullptr;
        launch<double, 0>(p, grid, stream);
    }
    return {units_out, state_out, answer_out, every, saves};
}

std::vector<torch::Tensor> segment_backward(torch::Tensor weights, std::vector<int64_t> offsets,
                                            torch::Tensor peers, torch::Tensor inverse,
                                            torch::Tensor masks, torch::Tensor rdeg,
                                            torch::Tensor saves, torch::Tensor units,
                                            torch::Tensor clue, torch::Tensor danswers,
                                            int64_t cycles, int64_t rounds, int64_t mode,
                                            torch::Tensor timing);

std::vector<int64_t> timing() {
    int n = 0;
    cudaMemcpyFromSymbol(&n, g_nsamples, sizeof(n));
    std::vector<long long> host(n > 0 ? n : 0);
    if (n > 0) cudaMemcpyFromSymbol(host.data(), g_timing, sizeof(long long) * n);
    int zero = 0;
    cudaMemcpyToSymbol(g_nsamples, &zero, sizeof(zero));
    return std::vector<int64_t>(host.begin(), host.end());
}

//: The build's layout, for the host's shared-memory formulas to be checked
//: against: the widths and the graph, the strides, the region rows, the two
//: kernels' shared-memory footprints in bytes (float), and the modes built.
std::vector<int64_t> layout() {
    return {KDIM, KSD, KHID, KYD, NC, DEG, MS, SDP, HIDP, YDP, DIMP, CAK, ROWS, NTHREADS,
            WSS, WS(KHID), WSY, WS(KDIM), WS(CAK), NCP, PC_T, NTILES_N, LPC, CGROUPS,
            (int64_t)ROUND_REC, (int64_t)CYC_REC, smem_forward(), smem_backward(),
            KPADDED, KSYM, KMODES, KLAYERS};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segment", &segment, "forward segment");
    m.def("segment_backward", &segment_backward, "backward of a saved forward segment");
    m.def("layout", &layout, "compile-time layout constants");
    m.def("modes", []() { return (int64_t)KMODES; }, "the float GEMM modes this build has");
    m.def("timing", &timing, "phase cycle counts of block 0 (KTIMING builds), then reset");
}

// The backward of a saved forward segment: one board per block (a
// persistent block takes several boards in a grid-stride loop), the
// round's gradients on chip, the weight gradients accumulated per BOARD
// (zeroed at the top of every board, not once per block) in an
// L2-resident copy of the packed layout, kept separate per board and
// summed on the host only after the per-trajectory clip has scaled each
// row (see autograd.py's ``SegmentFunction.backward``).
// The order per cycle, backwards: the refresh, then the rounds (LayerNorm ->
// GRU -> the three GEMMs' transposes and weight gradients -> the pool), then
// the per-cycle constant's gradient into the clue and the entering answer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"

template <typename T>
struct BParams {
    const T* __restrict__ weights;
    Graph<T> graph;
    const T* __restrict__ saves;        // the forward's record, B x board_record
    const T* __restrict__ units_in;     // (B, NC, DIM): the entry round's beliefs
    const T* __restrict__ clue_in;      // (B, NC, DIM)
    const T* __restrict__ danswers;     // (cycles, B, NC, YD): d loss / d answer after each cycle
    T* __restrict__ dpacked;            // (B, off.total): per-board weight gradients
    T* __restrict__ dclue;              // (B, NC, DIM)
    T* __restrict__ scratch;            // fp64 path
    long long* __restrict__ timing;     // KTIMING: [0] = count, then (phase << 40) | cycles (nullptr: off)
    int B, cycles, rounds;
    Offsets off;
};

// the backward's phase timing goes through an explicit buffer (each translation
// unit would otherwise own a copy of the header's globals)
#ifdef KTIMING
#define BPHASE(i) do { if (p.timing && blockIdx.x == 0 && threadIdx.x == 0) { long long now_ = clock64(); \
        int at_ = (int)p.timing[0]; if (at_ < NSAMPLES) { p.timing[1 + at_] = ((long long)(i) << 40) | (now_ - tlast_); p.timing[0] = at_ + 1; } \
        tlast_ = now_; } } while (0)
#else
#define BPHASE(i) do { } while (0)
#endif

// The shared-memory regions (rows of MS) are laid out in common.cuh, which the
// host's ``smem_bytes`` replicates.

template <typename T, int MODE>
__global__ void __launch_bounds__(NTHREADS, 1) segment_bwd_kernel(BParams<T> p) {
    long long tlast_ = 0; (void)tlast_;
#ifdef KTIMING
    tlast_ = clock64();
#endif
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr bool ONCHIP = sizeof(T) == 4;
    constexpr int SMEM_T = BROWS * MS + (ONCHIP ? W1A_FLOATS + KLAYERS * KSD + KYD : 0);
    T* base = ONCHIP ? reinterpret_cast<T*>(smem_raw) : p.scratch + (size_t)blockIdx.x * SMEM_T;
    T* state = base;                        // [SDP][MS], row KSD = 1 (the bias column)
    T* dstate = state + SDP * MS;           // [SDP][MS]
    T* dca = dstate + SDP * MS;             // [CAK][MS]: d clue rows [0, DIMP), d answer rows [DIMP, CAK)
    T* dclue = dca;
    T* dans = dca + DIMP * MS;
    T* ca = dca + CAK * MS;                 // [CAK][MS]: the clue, and the answer entering the cycle
    T* answer_in = ca + DIMP * MS;
    T* ra = ca + CAK * MS;                  // [BA_ROWS][MS]
    T* rb = ra + BA_ROWS * MS;              // [BB_ROWS][MS]
    T* rc = rb + BB_ROWS * MS;              // [BC_ROWS][MS]
    const Offsets o = p.off;
    // one packed buffer per layer, o.total apart, and the same layout of gradients;
    // w1a is resident (one layer at a time: restaged per application when KLAYERS > 1),
    // the state LayerNorm gains of every layer and the refresh's sit after it
    const T* w1a = ONCHIP ? rc + BC_ROWS * MS : p.weights + o.w1a;
    const T* lns_g = ONCHIP ? w1a + W1A_FLOATS : p.weights + o.lns_g;
    const T* lny_g = ONCHIP ? lns_g + KLAYERS * KSD : p.weights + o.lny_g;
    for (int i = threadIdx.x; i < BROWS * MS; i += NTHREADS) base[i] = T(0);
    if (ONCHIP) {
        if (KLAYERS == 1) copy_words<T>(const_cast<T*>(w1a), p.weights + o.w1a, W1A_FLOATS);
        for (int l = 0; l < KLAYERS; ++l)
            for (int i = threadIdx.x; i < KSD; i += NTHREADS)
                const_cast<T*>(lns_g)[l * KSD + i] = p.weights[(size_t)l * o.total + o.lns_g + i];
        for (int i = threadIdx.x; i < KYD; i += NTHREADS) const_cast<T*>(lny_g)[i] = p.weights[o.lny_g + i];
    }
    Ports<T> ports;
    load_ports<T>(ports, p.graph);
    constexpr int YITEMS = (KYD * NC + NTHREADS - 1) / NTHREADS;
    // the staged per-cycle maps (region A, above the refresh's answer rows)
    T* wr_s = ONCHIP ? ra + 2 * YDP * MS : const_cast<T*>(p.weights + o.wrih);
    T* wrh_s = ONCHIP ? wr_s + (o.wrhh - o.wrih) : const_cast<T*>(p.weights + o.wrhh);
    T* whh_s = ONCHIP ? ra : const_cast<T*>(p.weights + o.whh);
    T* w2p_s = ONCHIP ? rc : const_cast<T*>(p.weights + o.w2p);
    T* wca_s = ONCHIP ? ra : const_cast<T*>(p.weights + o.wca);   // staged below the hidden n-gate rows
    __syncthreads();

    for (int b = blockIdx.x; b < p.B; b += gridDim.x) {
        // this board's own row of dpacked: zeroed here (not once, outside the
        // loop) so a persistent block's boards do not accumulate into one
        // another -- the caller sums over boards in Python, so every board's
        // parameter-gradient contribution must come back separately (the
        // per-trajectory clip needs it; see autograd.py)
        T* dw = p.dpacked + (size_t)b * (KLAYERS * o.total);
        for (int i = threadIdx.x; i < KLAYERS * o.total; i += NTHREADS) dw[i] = T(0);
        const T* rec = p.saves + (size_t)b * board_record(p.cycles, p.rounds * KLAYERS);
        const T* crec = rec + (size_t)p.cycles * p.rounds * KLAYERS * ROUND_REC;
        zero_rows<T>(dstate, SDP); zero_rows<T>(dca, CAK);
        load_T<T, KDIM, DIMP>(ca, p.clue_in + (size_t)b * NC * KDIM);
        __syncthreads();
        fill_row<T>(ca + (CAK - 1) * MS, T(1));                        // the bias column of wca
        __syncthreads();
        BPHASE(20);
        for (int cyc = p.cycles - 1; cyc >= 0; --cyc) {
            // -- the refresh of cycle cyc ------------------------------------------
            const T* ck = crec + (size_t)cyc * CYC_REC;
            {
                // dans: the chain so far (from cycle cyc + 1) plus this cycle's readout
                const T* dy = p.danswers + ((size_t)cyc * p.B + b) * NC * KYD;
                for (int i = threadIdx.x; i < KYD * NC; i += NTHREADS)
                    dans[(i % KYD) * MS + i / KYD] += dy[i];
                load_rows<T>(ra, ck + CSAVE_ANS, KYD);                 // answer entering the refresh
                load_rows<T>(answer_in, ck + CSAVE_ANS, KYD);          // ... also the rounds' gconst input
                load_rows<T>(ra + YDP * MS, ck + CSAVE_GHN, KYD);      // its hidden n-gate
                load_rows<T>(rc, ck + CSAVE_RZN, 3 * KYD);             // r | z | n
                // the state entering the refresh: the next round's entering state, or the final one
                load_rows<T>(state, cyc + 1 < p.cycles
                             ? rec + (size_t)(cyc + 1) * p.rounds * KLAYERS * ROUND_REC + SAVE_STATE
                             : crec + (size_t)p.cycles * CYC_REC, KSD);
                if (ONCHIP) copy_words<T>(wr_s, p.weights + o.wrih, o.we - o.wrih);
                __syncthreads();
                fill_row<T>(state + KSD * MS, T(1));                   // the bias columns
                fill_row<T>(ra + KYD * MS, T(1));
                __syncthreads();
                BPHASE(31);
                ln_bwd<T, KYD>(dans, ra, rc, lny_g, rb);
                __syncthreads();
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS>(dw + o.lnyw, rb, state, 2 * KYD, SDP / 8);   // sums into column KSD
                __syncthreads();
                gru_bwd<T, KYD, YITEMS>(dans, ra, rc, ra + YDP * MS, rb, nullptr);
                __syncthreads();
                BPHASE(32);
                // dgr in rb (3Y rows), dgrh in rc (3Y rows), dans = dh' z
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS>(dw + o.wrih, rb, state, N3Y, SDP / 8);
                gemm_dw<T, MODE, CMIN(NTILES_N, 3), WSY>(dw + o.wrhh, rc, ra, N3Y, YDP / 8);
                gemm_at<T, MODE, RUP(N3Y, 8), WSS, CMIN(NTILES_N, 4), true>(dstate, rb, wr_s, KSD);
                gemm_at<T, MODE, RUP(N3Y, 8), WSY, CMIN(NTILES_N, 3), true>(dans, rc, wrh_s, KYD);
                __syncthreads();
                BPHASE(33);
            }
            // -- the layer applications of cycle cyc, last first ---------------------
            for (int a = p.rounds * KLAYERS - 1; a >= 0; --a) {
                const int l = a % KLAYERS, k = cyc * p.rounds * KLAYERS + a;
                const T* wg = p.weights + (size_t)l * o.total;     // layer l's weights ...
                T* dwl = dw + (size_t)l * o.total;                 // ... and their gradients
                const T* rk = rec + (size_t)k * ROUND_REC;
                load_rows<T>(state, rk + SAVE_STATE, KSD);
                load_rows<T>(rc, rk + SAVE_RZN, 3 * KSD);
                load_rows<T>(ra + N3S * MS, rk + SAVE_GHN, KSD);   // gh_n above the gates
                if (ONCHIP) copy_words<T>(whh_s, wg + o.whh, o.bhh - o.whh);
                if constexpr (ONCHIP && KLAYERS > 1)               // w1a: this layer's, used last
                    copy_words<T>(const_cast<T*>(w1a), wg + o.w1a, W1A_FLOATS);
                const T* whh_l = ONCHIP ? whh_s : wg + o.whh;
                const T* wca_l = ONCHIP ? wca_s : wg + o.wca;
                const T* w2p_l = ONCHIP ? w2p_s : wg + o.w2p;
                const T* w1a_l = ONCHIP ? w1a : wg + o.w1a;
                const T* lns_l = ONCHIP ? lns_g + l * KSD : wg + o.lns_g;
                __syncthreads();
                fill_row<T>(state + KSD * MS, T(1));
                __syncthreads();
                BPHASE(21);
                ln_bwd<T, KSD>(dstate, state, rc, lns_l, rb);
                __syncthreads();
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS>(dwl + o.lnw, rb, state, 2 * KSD, SDP / 8);   // sums into column KSD
                __syncthreads();
                BPHASE(22);
                gru_bwd<T, KSD, GC_ITEMS>(dstate, state, rc, ra + N3S * MS, rb, nullptr);
                __syncthreads();
                BPHASE(23);
                // dgi in rb, dgh in rc
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS>(dwl + o.whh, rc, state, N3S, SDP / 8);
                gemm_at<T, MODE, RUP(N3S, 8), WSS, CMIN(NTILES_N, 4), true>(dstate, rc, whh_l, KSD);
                __syncthreads();
                BPHASE(24);
                // the per-cycle constant's gradient, per application: dW_ca += dgi . [clue; answer]^T,
                // d[clue; answer] += Wca^T dgi (the clue's into the encoder, the answer's into
                // the chain); wca staged into region A (whh and gh_n are dead)
                if (ONCHIP) copy_words<T>(wca_s, wg + o.wca, o.bca - o.wca);
                __syncthreads();
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WS(CAK)>(dwl + o.wca, rb, ca, N3S, CAK / 8);
                gemm_at<T, MODE, RUP(N3S, 8), WS(CAK), CMIN(NTILES_N, 4), true>(dca, rb, wca_l, CAK);
                __syncthreads();
                BPHASE(34);
                if (ONCHIP) copy_words<T>(w2p_s, wg + o.w2p, o.whh - o.w2p);
                load_rows<T>(ra + PCELL_OFF * MS, rk + SAVE_POOLED, KHID);
                __syncthreads();
                BPHASE(25);
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WS(KHID)>(dwl + o.w2p, rb, ra + PCELL_OFF * MS, N3S, HIDP / 8);
                gemm_at<T, MODE, RUP(N3S, 8), WS(KHID), CMIN(NTILES_N, 6), false, false, true>(ra, rb, w2p_l, KHID);   // dpooled, cell-major
                __syncthreads();
                BPHASE(26);
                // shared (feature-major in the record) -> cell-major in region C via region B;
                // per_cell cell-major into region B afterwards
                load_rows<T>(rb, rk + SAVE_SHARED, KHID);
                if (k == 0) load_T<T, KDIM, DIMP>(rc + RUP(CM_ROWS, 4) * MS, p.units_in + (size_t)b * NC * KDIM);
                __syncthreads();
                to_cell_major<T>(rc, rb, KHID);
                __syncthreads();
                load_block<T>(rb, rk + SAVE_PCELL, NC * PC_T);
                __syncthreads();
                BPHASE(27);
                pool_bwd_shared<T>(ra + PCELL_OFF * MS, rc, rb, ra, ports);   // dshared (feature-major) over the dead pooled
                __syncthreads();
                BPHASE(28);
                pool_bwd_pcell<T>(rb, rc, ra, ports);                     // dper_cell, cell-major, in place
                __syncthreads();
                BPHASE(29);
                gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS>(dwl + o.w1a, ra + PCELL_OFF * MS, state, KHID, SDP / 8);
                gemm_at<T, MODE, HIDP, WSS, CMIN(NTILES_N, 4), true>(dstate, ra + PCELL_OFF * MS, w1a_l, KSD);
                if (k > 0) {
                    gemm_dw<T, MODE, CMIN(NTILES_N, 5), WSS, true>(dwl + o.w1a + (size_t)HIDP * WSS, rb, state, KHID, SDP / 8);
                    gemm_at<T, MODE, HIDP, WSS, CMIN(NTILES_N, 4), true, true>(dstate, rb, w1a_l + (size_t)HIDP * WSS, KSD);
                } else {
                    gemm_dw<T, MODE, CMIN(NTILES_N, 2), WS(KDIM), true>(dwl + o.wp, rb, rc + RUP(CM_ROWS, 4) * MS, KHID, DIMP / 8);
                }
                __syncthreads();
                BPHASE(30);
            }
        }
        // the board's clue gradient out
        store_T<T, KDIM>(p.dclue + (size_t)b * NC * KDIM, dclue);
        __syncthreads();
    }
}

static Offsets offsets_of_(const std::vector<int64_t>& v) {
    Offsets o;
    int* f = reinterpret_cast<int*>(&o);
    TORCH_CHECK(v.size() == sizeof(Offsets) / sizeof(int), "offsets: wrong length");
    for (size_t i = 0; i < v.size(); ++i) f[i] = (int)v[i];
    return o;
}

template <typename T, int MODE>
static void launch_bwd(BParams<T> p, int grid, cudaStream_t stream) {
    constexpr size_t smem = sizeof(T) == 4
        ? (size_t)(BROWS * MS + W1A_FLOATS + KLAYERS * KSD + KYD) * sizeof(T) * 1 : 0;
    if (sizeof(T) == 4) {
        static bool configured = false;
        if (!configured) {
            TORCH_CHECK(smem <= 227 * 1024, "segment backward: shared memory ", smem, " bytes");
            cudaFuncSetAttribute(segment_bwd_kernel<T, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            configured = true;
        }
    }
    segment_bwd_kernel<T, MODE><<<grid, NTHREADS, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> segment_backward(torch::Tensor weights, std::vector<int64_t> offsets,
                                            torch::Tensor peers, torch::Tensor inverse,
                                            torch::Tensor masks, torch::Tensor rdeg,
                                            torch::Tensor saves, torch::Tensor units,
                                            torch::Tensor clue, torch::Tensor danswers,
                                            int64_t cycles, int64_t rounds, int64_t mode,
                                            torch::Tensor timing) {
    const Wiring g = wiring_of(peers, inverse, masks, rdeg);
    long long* timing_ptr = timing.numel() > 0 ? reinterpret_cast<long long*>(timing.data_ptr<int64_t>()) : nullptr;
    TORCH_CHECK(saves.is_cuda() && danswers.is_cuda() && units.is_cuda() && clue.is_cuda());
    const c10::cuda::OptionalCUDAGuard guard(saves.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int B = (int)units.size(0);
    TORCH_CHECK(danswers.sizes() == torch::IntArrayRef({cycles, B, NC, KYD}), "danswers shape");
    units = units.contiguous(); clue = clue.contiguous(); danswers = danswers.contiguous();
    weights = weights.contiguous(); saves = saves.contiguous();
    const Offsets off = offsets_of_(offsets);
    TORCH_CHECK(weights.numel() == (int64_t)KLAYERS * off.total, "weights: ", weights.numel(),
                " elements, the build has ", KLAYERS, " layer(s) of ", off.total);
    static int n_sms = 0;
    if (n_sms == 0) cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, saves.device().index());
    const int grid = B < 1 ? 1 : (B < n_sms ? B : n_sms);
    // one row per board (not per block): a persistent block processes many
    // boards in its grid-stride loop, and the caller needs each board's own
    // parameter-gradient contribution separately for the per-trajectory clip
    // (autograd.py), summing them itself once the clip has scaled each row
    auto dpacked = torch::empty({B, (int64_t)KLAYERS * off.total}, saves.options());
    auto dclue = torch::empty({B, NC, KDIM}, saves.options());
    if (B > 0 && saves.dtype() == torch::kFloat32) {
        BParams<float> p;
        p.weights = weights.data_ptr<float>(); p.graph = graph_of<float>(g);
        p.saves = saves.data_ptr<float>(); p.units_in = units.data_ptr<float>();
        p.clue_in = clue.data_ptr<float>(); p.danswers = danswers.data_ptr<float>();
        p.dpacked = dpacked.data_ptr<float>(); p.dclue = dclue.data_ptr<float>();
        p.scratch = nullptr; p.timing = timing_ptr; p.B = B; p.cycles = (int)cycles; p.rounds = (int)rounds; p.off = off;
        MODE_DISPATCH(mode, launch_bwd<float, MODE_>(p, grid, stream));
    } else if (B > 0) {
        TORCH_CHECK(saves.dtype() == torch::kFloat64, "float32 or float64");
        auto scratch = torch::zeros({(int64_t)grid * (BROWS * MS)}, saves.options());
        BParams<double> p;
        p.weights = weights.data_ptr<double>(); p.graph = graph_of<double>(g);
        p.saves = saves.data_ptr<double>(); p.units_in = units.data_ptr<double>();
        p.clue_in = clue.data_ptr<double>(); p.danswers = danswers.data_ptr<double>();
        p.dpacked = dpacked.data_ptr<double>(); p.dclue = dclue.data_ptr<double>();
        p.scratch = scratch.data_ptr<double>(); p.timing = timing_ptr; p.B = B; p.cycles = (int)cycles; p.rounds = (int)rounds; p.off = off;
        launch_bwd<double, 0>(p, grid, stream);
    }
    return {dpacked, dclue};   // unsummed: per board, (B, KLAYERS * off.total)
}

// Forward-only segment kernel for the peer-graph sudoku model.
//
// One thread block owns one board for a whole segment: ``cycles`` cycles of
// ``rounds`` rounds of peer message passing, each cycle closed by the answer
// refresh.  Every activation lives in shared memory, laid out feature-major
// (``[feature][cell]``, the cell index fastest, row stride MS) so that a warp
// whose lanes are consecutive cells reads and writes conflict-free, and so do
// the ``mma`` fragment loads.  The weights are packed once on the host
// (compositions folded, rows padded) and copied to shared memory once per
// block; the per-cycle gate constant lives in registers.
//
// The maps of a round, per cell (see discopy.neural.cell.PeerSite.forward):
//   shared   = Ws state + b0                          (G1a, rows [0, HID))
//   per_cell = Wp belief = (Wp We) state + Wp be      (G1a, rows [HID, 2HID);
//                                                      from the given beliefs
//                                                      on the entry round)
//   pooled   = mean_p relu(shared + per_cell[peer_p])
//   gi       = (Wi_pool W2) pooled + gconst           (G2; gconst = the clue and
//                                                      answer gates + all input
//                                                      biases, once per cycle)
//   gh       = Whh state + bhh                        (G1b)
//   state'   = LayerNorm(GRU(gi, gh, state))
// and of the refresh: answer' = LayerNorm(GRU(Wr_ih state + b, Wr_hh answer + b, answer)).
//
// GEMMs: float runs on the tensor cores as ``mma.m16n8k8`` tf32 with the
// 3xTF32 split (MODE 1, fp32-accurate) or single tf32 (MODE 2); MODE 0 and
// the float64 instantiation run the same products as plain FMAs.  In the
// mma the *weights* are the row operand (16 outputs) and the cells the
// column operand (8 cells), so the 81 cells pad to 88, not 96.
//
// The graph is a build parameter, not the sudoku clique: KNC cells of KDEG
// ports each, the port -> cell table read at run time.  KPADDED graphs (cells
// of unequal degree) carry a per-cell validity mask and divide the pool by the
// cell's real degree; on a KSYM graph (peers(c) symmetric, sudoku) the
// backward's inverse relation *is* the peer table and is never read.  Both are
// compile-time, so an unpadded symmetric build is the code that was written
// for sudoku.
//
// Compile-time widths: KDIM, KSD (state), KHID, KYD; graph: KNC, KDEG,
// KPADDED, KSYM; precision: KMODES (a bitmask of the float GEMM modes built).

#pragma once
#include <cuda_runtime.h>

#ifndef KDIM
#define KDIM 15
#define KSD 39
#define KHID 76
#define KYD 20
#endif
#ifndef KMODE
#define KMODE 1
#endif
#ifndef KNC
#define KNC 81
#endif
#ifndef KDEG
#define KDEG 20
#endif
#ifndef KPADDED
#define KPADDED 0                // 1: cells of unequal degree (a port validity mask, a real divisor)
#endif
#ifndef KSYM
#define KSYM 1                   // 1: p in peers(c) <=> c in peers(p) (the inverse table is the peer table)
#endif
#ifndef KMODES
#define KMODES 7                 // the float GEMM modes built: bit 0 fma, bit 1 fp32 (3xTF32), bit 2 tf32
#endif
#ifndef KLAYERS
#define KLAYERS 1                // distinct cell layers per round, one packed buffer each (see packing.py)
#endif

#define RUP(x, m) (((x) + (m) - 1) / (m) * (m))
#define CMAX(a, b) ((a) > (b) ? (a) : (b))
#define CMIN(a, b) ((a) < (b) ? (a) : (b))
#define NC KNC
#define DEG KDEG
#define NTILES_N (RUP(NC, 8) / 8)          // cell tiles of 8 (sudoku: 11)
#ifndef MS
// cell stride: the smallest multiple of 8 at or above NC (a tile past the last
// one spills into the next row) that is 8 mod 16, which keeps the cells as
// columns conflict-free and the cells as the contraction 4 mod 8.  81 -> 88.
#define MS (RUP(NC, 8) + (RUP(NC, 8) % 16 == 8 ? 0 : 8))
#endif
static_assert(MS >= 8 * NTILES_N && MS % 8 == 0, "cell stride");
#define NTHREADS 256             // 8 warps: 255 registers per thread, no spills
#define NWARPS (NTHREADS / 32)
#define WS(k) (RUP(k, 8) + 4)    // weight row stride: K padded to 8, then == 4 mod 8
// The state and the answer are padded to a row *past* their width: that row
// carries the ones the backward contracts against for the bias gradients, and
// the weight stride of a map on them has to hold it (39 -> 40 rows, stride 44,
// as before; a width that is a multiple of 8 is what needs the + 1).
#define SDP RUP(KSD + 1, 8)      // state rows
#define YDP RUP(KYD + 1, 8)      // answer rows
#define WSS (SDP + 4)            // weight row stride of a map on the state
#define WSY (YDP + 4)            // ... and on the answer
#define HIDP RUP(KHID, 8)        // pooled rows
#define DIMP RUP(KDIM, 8)
#define CAK (DIMP + YDP)         // the clue|answer buffer, the input of gconst
#define N1A (2 * HIDP)           // shared (padded to HIDP rows) | per_cell
#define PC_T (HIDP + 4)          // per_cell is stored cell-major, [NC][PC_T], from row HIDP of region A;
                                 // the stride == 4 mod 8 keeps the 16-byte gathers off a common bank group
#define PCELL_ROWS ((NC * PC_T + MS - 1) / MS)   // the rows that cell-major block occupies
#define N3S (3 * KSD)            // GRU gates
#define N3Y (3 * KYD)
#define DEG4 ((DEG + 3) / 4)     // the packed peer table: four ports per word
#define NGF CMIN(NTILES_N, 11)   // cell tiles per warp on the forward's tensor-core GEMMs
// The per-cycle weight blocks are staged through the activation regions, so
// every region is at least as tall as what is staged in it (few cells: a short
// row, and a weight matrix needs more rows than any activation).
#define WP_ROWS ((RUP(KHID, 16) * WS(KDIM) + MS - 1) / MS)
#define WCA_ROWS ((RUP(N3S, 16) * WS(CAK) + MS - 1) / MS)
#define WR_ROWS ((RUP(N3Y, 16) * (WSS + WSY) + MS - 1) / MS)
#define WE_ROWS ((RUP(KDIM, 16) * WSS + MS - 1) / MS)
#define WHH_ROWS ((RUP(N3S, 16) * WSS + MS - 1) / MS)
#define W2P_ROWS ((RUP(N3S, 16) * WS(KHID) + MS - 1) / MS)
#define RA_ROWS CMAX(CMAX(N1A, HIDP + PCELL_ROWS), CMAX(N3S, RUP(N3Y, 16) + WR_ROWS))   // region A
#define RB_ROWS CMAX(CMAX(HIDP, N3S), CMAX(CMAX(N3Y, WE_ROWS), DIMP + WP_ROWS + WCA_ROWS))  // region B
#define ROWS (SDP + CAK + RA_ROWS + RB_ROWS)
#define STAT_FLOATS (2 * MS)
#define GC_ITEMS ((KSD * NC + NTHREADS - 1) / NTHREADS)   // GRU items per thread

// The backward's shared-memory regions (rows of MS), see segment_bwd.cu.
#define PCELL_OFF CMAX(HIDP, PCELL_ROWS)                  // the pooled rows, above the cell-major dpooled
// r|z|n + gh_n, or dpooled + pooled; whh, wca and the refresh maps staged
#define BA_ROWS CMAX(CMAX(N3S + KSD, PCELL_OFF + KHID), CMAX(CMAX(WHH_ROWS, WCA_ROWS), 2 * YDP + WR_ROWS))
#define CM_ROWS PCELL_ROWS                                // a cell-major [NC][PC_T] block
#define BB_ROWS CMAX(N3S, CM_ROWS)                        // dgi, then per_cell (cell-major)
// r|z|n -> dgh, staged w2p and wca, shared (cell-major) + units
#define BC_ROWS CMAX(CMAX(N3S, W2P_ROWS), CMAX(RUP(CM_ROWS, 4) + DIMP + 1, WCA_ROWS))
#define BROWS (SDP + SDP + CAK + CAK + BA_ROWS + BB_ROWS + BC_ROWS)   // + dclue|dans and clue|answer buffers
#define W1A_FLOATS (RUP(N1A, 16) * WSS)

//: The two kernels' shared-memory footprints in bytes for a float build; the
//: forward's grows by the resident weights, which the host knows.  ``build.py``
//: replicates both and ``supports`` refuses a build that will not fit.
__host__ __device__ inline long long smem_forward() {
    return (long long)(ROWS * MS + STAT_FLOATS) * sizeof(float);
}
__host__ __device__ inline long long smem_backward() {
    return (long long)(BROWS * MS + W1A_FLOATS + KLAYERS * KSD + KYD) * sizeof(float);
}

// Only the requested modes are instantiated: one mode is a third of the
// compile time of all three.  MODE_ is the mode as a compile-time constant.
#if (KMODES & 1)
#define MODE_CASE_0(...) do { constexpr int MODE_ = 0; __VA_ARGS__; } while (0)
#else
#define MODE_CASE_0(...) TORCH_CHECK(false, "the 'fma' mode is not in this build")
#endif
#if (KMODES & 2)
#define MODE_CASE_1(...) do { constexpr int MODE_ = 1; __VA_ARGS__; } while (0)
#else
#define MODE_CASE_1(...) TORCH_CHECK(false, "the 'fp32' mode is not in this build")
#endif
#if (KMODES & 4)
#define MODE_CASE_2(...) do { constexpr int MODE_ = 2; __VA_ARGS__; } while (0)
#else
#define MODE_CASE_2(...) TORCH_CHECK(false, "the 'tf32' mode is not in this build")
#endif
#define MODE_DISPATCH(mode, ...) do {                  \
    if ((mode) == 0) MODE_CASE_0(__VA_ARGS__);         \
    else if ((mode) == 2) MODE_CASE_2(__VA_ARGS__);    \
    else MODE_CASE_1(__VA_ARGS__);                     \
} while (0)

// Packed weight offsets (in elements), fixed by the host packer in the same order.
struct Offsets {
    int w1a, b1a, w2p, whh, bhh, brih, brhh, be,
        lns_g, lns_b, lny_g, lny_b, resident,      // [0, resident) is copied to smem
        wrih, wrhh, we, wp, wca, bca,              // staged through region A/B when used
        lnw, lnyw,                                 // backward only: LayerNorm gradient accumulators
        total;
};

// The wiring, one build's worth: the port -> cell table the forward gathers
// through, the backward's inverse relation (KSYM: the same table, never read),
// and -- KPADDED only -- the packed port validity bits of both and the real
// degree the pool divides by.
template <typename T>
struct Graph {
    const int* __restrict__ peers;                  // (NC, DEG)
    const int* __restrict__ inverse;                // (NC, DEG): the cells p is a peer of
    const unsigned long long* __restrict__ masks;   // (NC, 2): peer bits, then inverse bits
    const T* __restrict__ rdeg;                     // (NC, ): the real degree
};

//: One cell's ports in registers: a byte per peer (NC <= 256), so a gather is
//: a shift and a mask rather than a load.  The inverse table and the validity
//: bits collapse to nothing on a symmetric unpadded build.
template <typename T>
struct Ports {
    unsigned pe[DEG4];
    unsigned inv[KSYM ? 1 : DEG4];
    unsigned long long valid, ivalid;
    T rdeg;
};

template <typename T>
__device__ __forceinline__ void load_ports(Ports<T>& q, const Graph<T>& g) {
    const int c = threadIdx.x % NC;
#pragma unroll
    for (int j = 0; j < DEG4; ++j) {
        unsigned v = 0, w = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (j * 4 + i < DEG) {
                v |= (unsigned)__ldg(g.peers + c * DEG + j * 4 + i) << (8 * i);
                if (!KSYM) w |= (unsigned)__ldg(g.inverse + c * DEG + j * 4 + i) << (8 * i);
            }
        q.pe[j] = v;
        if (!KSYM) q.inv[j] = w;
    }
    q.valid = q.ivalid = ~0ull;
    q.rdeg = T(DEG);
    if (KPADDED) {
        q.valid = __ldg(g.masks + 2 * c);
        q.ivalid = __ldg(g.masks + 2 * c + 1);
        q.rdeg = g.rdeg[c];
    }
}

//: The cell behind port ``q`` of the peer (``inv``: the inverse) table.
#define PORT(ports, q) (int)((ports.pe[(q) / 4] >> (8 * ((q) % 4))) & 0xff)
#define IPORT(ports, q) (int)((KSYM ? ports.pe[(q) / 4] : ports.inv[(q) / 4]) >> (8 * ((q) % 4)) & 0xff)

template <typename T>
struct Params {
    const T* __restrict__ weights;
    Graph<T> graph;
    const T* __restrict__ units_in;     // (B, NC, DIM)
    const T* __restrict__ state_in;     // (B, NC, SD)
    const T* __restrict__ clue_in;      // (B, NC, DIM)
    const T* __restrict__ answer_in;    // (B, NC, YD)
    T* __restrict__ units_out;
    T* __restrict__ state_out;
    T* __restrict__ answer_out;
    T* __restrict__ every;              // (cycles, B, NC, YD)
    T* __restrict__ scratch;            // fp64 path: per-block activation storage
    T* __restrict__ saves;              // training: per-board activation record (nullptr: none)
    int B, cycles, rounds;
    Offsets off;
};

// The activation record the backward reads, per board, compact [rows][NC]:
// one ROUND_REC per global layer application k (the state entering it, the
// port encodings, the pooled vector, the gates r|z|n and the hidden n-gate;
// a round is KLAYERS of them, so the callers pass rounds * KLAYERS), one
// CYC_REC per cycle (the answer entering the refresh and its gates), then
// the final state and answer.
#define NCP RUP(NC, 4)           // a record row: NC floats padded to a 16-byte multiple
#define SAVE_STATE 0
#define SAVE_SHARED (SAVE_STATE + KSD * NCP)
#define SAVE_PCELL (SAVE_SHARED + KHID * NCP)           // cell-major [NC][PC_T], as on chip
#define SAVE_POOLED (SAVE_PCELL + NC * PC_T)
#define SAVE_RZN (SAVE_POOLED + KHID * NCP)
#define SAVE_GHN (SAVE_RZN + 3 * KSD * NCP)
#define ROUND_REC (SAVE_GHN + KSD * NCP)
#define CSAVE_ANS 0
#define CSAVE_RZN (CSAVE_ANS + KYD * NCP)
#define CSAVE_GHN (CSAVE_RZN + 3 * KYD * NCP)
#define CYC_REC (CSAVE_GHN + KYD * NCP)
static_assert(NC * PC_T % 4 == 0 && ROUND_REC % 4 == 0 && CYC_REC % 4 == 0, "record blocks stay 16-byte aligned");
__host__ __device__ inline long long board_record(int cycles, int rounds) {
    return (long long)cycles * rounds * ROUND_REC + (long long)cycles * CYC_REC + (KSD + KYD) * NCP;
}

template <typename T> struct Vec4;
template <> struct Vec4<float> {
    typedef float4 t;
    static __device__ __forceinline__ float4 load(const float* p) {
        return *reinterpret_cast<const float4*>(p);
    }
};
template <> struct Vec4<double> {
    typedef double4 t;
    static __device__ __forceinline__ double4 load(const double* p) {
        double2 a = *reinterpret_cast<const double2*>(p);
        double2 b = *reinterpret_cast<const double2*>(p + 2);
        return make_double4(a.x, a.y, b.x, b.y);
    }
};
template <typename T>
__device__ __forceinline__ typename Vec4<T>::t make_vec4(T a, T b, T c, T d) {
    typename Vec4<T>::t v;
    v.x = a; v.y = b; v.z = c; v.w = d;
    return v;
}

#ifndef KFAST
#define KFAST 1                  // float: MUFU-based exp / reciprocal (2 ulp) in the gates
#endif
// global -> shared 16-byte asynchronous copy, and the wait for a thread's copies
// L2 cache policies: the activation record streams through once (evict first), the
// per-block weight-gradient accumulators are hot every round (evict last).
__device__ __forceinline__ unsigned long long policy_evict_first() {
    unsigned long long p;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ unsigned long long policy_evict_last() {
    unsigned long long p;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(gsrc));
}
__device__ __forceinline__ void cp_async16_stream(void* smem_dst, const void* gsrc, unsigned long long pol) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" :: "r"(d), "l"(gsrc), "l"(pol));
}
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ float2 ld_hot2(const float* p, unsigned long long pol) {
    float2 v;
    asm volatile("ld.global.L2::cache_hint.v2.f32 {%0, %1}, [%2], %3;" : "=f"(v.x), "=f"(v.y) : "l"(p), "l"(pol));
    return v;
}
__device__ __forceinline__ void st_hot2(float* p, float2 v, unsigned long long pol) {
    asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(p), "f"(v.x), "f"(v.y), "l"(pol));
}
template <typename T> __device__ __forceinline__ void st_stream4(T* p, typename Vec4<T>::t v) { *reinterpret_cast<typename Vec4<T>::t*>(p) = v; }
template <> __device__ __forceinline__ void st_stream4<float>(float* p, float4 v) { __stcs(reinterpret_cast<float4*>(p), v); }

template <typename T> __device__ __forceinline__ T sigmoid_(T x) { return T(1) / (T(1) + exp(-x)); }
template <typename T> __device__ __forceinline__ T tanh_(T x) { return tanh(x); }
#if KFAST
template <> __device__ __forceinline__ float sigmoid_<float>(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
template <> __device__ __forceinline__ float tanh_<float>(float x) { return 1.0f - __fdividef(2.0f, 1.0f + __expf(2.0f * x)); }
#else
template <> __device__ __forceinline__ float sigmoid_<float>(float x) { return 1.0f / (1.0f + expf(-x)); }
template <> __device__ __forceinline__ float tanh_<float>(float x) { return tanhf(x); }
#endif
template <typename T> __device__ __forceinline__ T rsqrt_(T x) { return rsqrt(x); }
template <> __device__ __forceinline__ float rsqrt_<float>(float x) { return rsqrtf(x); }

// ---------------------------------------------------------------------------
// GEMM, FMA path: out[n][c] = sum_k W[n][k] in[k][c] + bias[n], n < N, c < NC.
// in: [KP][MS]; W: [N][KWS] (any memory), KP a multiple of 4; a thread owns one
// cell and NT consecutive outputs; W is read as float4, uniform over the warp.
#define CELLS 4                                   // cells per thread on the FMA path
#define CGROUPS ((NC + CELLS - 1) / CELLS)        // 81 cells: 21 groups, c, c + 21, c + 42, c + 63 (84 <= MS)
#define TPB (NTHREADS / CGROUPS)                  // output tiles per group so that CGROUPS x tiles ~ NTHREADS
static_assert(TPB >= 1, "too many cells for the FMA path's cell groups");
template <int N> struct Tile { static constexpr int NT = (N + TPB - 1) / TPB; };  // 81 cells: 12 x 21 = 252 of 256 threads

// ``split``: output rows n >= split are stored cell-major, ``out[split * MS + c * PC_T + (n - split)]``
// (the per_cell layout the pool gathers from), rows below it feature-major.
template <typename T, int NT, int KP, int KWS>
__device__ __forceinline__ void gemm_fma(T* __restrict__ out, const T* __restrict__ in,
                                         const T* __restrict__ W, const T* __restrict__ bias,
                                         int N, int warp0, int nwarps, int split) {
    typedef typename Vec4<T>::t V4;
    const int tiles = (N + NT - 1) / NT;
    const int tid = threadIdx.x - warp0 * 32, nthreads = nwarps * 32;
    if (tid < 0 || tid >= nthreads) return;
    for (int item = tid; item < CGROUPS * tiles; item += nthreads) {
        const int c = item % CGROUPS, n0 = (item / CGROUPS) * NT;
        T acc[CELLS][NT];
#pragma unroll
        for (int i = 0; i < CELLS; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[i][j] = T(0);
        const T* wrow = W + (size_t)n0 * KWS;
        const T* col = in + c;
#pragma unroll 1
        for (int k = 0; k < KP; k += 4) {
            T x[CELLS][4];
#pragma unroll
            for (int i = 0; i < CELLS; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) x[i][q] = col[(k + q) * MS + i * CGROUPS];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                V4 w = Vec4<T>::load(wrow + j * KWS + k);
#pragma unroll
                for (int i = 0; i < CELLS; ++i) {
                    acc[i][j] += w.x * x[i][0];
                    acc[i][j] += w.y * x[i][1];
                    acc[i][j] += w.z * x[i][2];
                    acc[i][j] += w.w * x[i][3];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int n = n0 + j;
            if (n < N) {
                const T b = bias ? bias[n] : T(0);
#pragma unroll
                for (int i = 0; i < CELLS; ++i) {
                    const int cc = c + i * CGROUPS;
                    if (cc < NC) {
                        if (n < split) out[n * MS + cc] = acc[i][j] + b;
                        else if (n - split < KHID) out[split * MS + cc * PC_T + (n - split)] = acc[i][j] + b;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM, tensor-core path (float only).  C[m][n] = sum_k A[m][k] B[k][n] with
// A = W (16 outputs x 8 k per fragment) and B = in (8 k x 8 cells):
//   a0 = W[m0+g][k0+t]  a1 = W[m0+g+8][k0+t]  a2 = W[m0+g][k0+t+4]  a3 = W[m0+g+8][k0+t+4]
//   b0 = in[k0+t][n0+g]  b1 = in[k0+t+4][n0+g]
//   c0 = out[m0+g][n0+2t]  c1 = ..[n0+2t+1]  c2 = out[m0+g+8][n0+2t]  c3 = ..[n0+2t+1]
// with g = lane / 4, t = lane % 4.  A warp owns one m-tile and NGN cell tiles.
// Round to the nearest tf32 (10-bit mantissa) by integer add-and-mask: two
// instructions, no infinity check (every value here is finite).  The mma reads
// only the high 19 bits of an operand, so what is passed as the *small* part of
// a split is truncated, not rounded: its error is 2^-11 relative to a term that
// is itself 2^-11 of the value.
__device__ __forceinline__ unsigned tf32_(float x) {
    return (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MODE, int KP, int KWS, int NGN>
__device__ __forceinline__ void gemm_mma(float* __restrict__ out, const float* __restrict__ in,
                                         const float* __restrict__ W, const float* __restrict__ bias,
                                         int N, int warp0, int nwarps, int split) {
    const int warp = (int)threadIdx.x / 32 - warp0;
    if (warp < 0 || warp >= nwarps) return;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    constexpr int NGROUPS = (NTILES_N + NGN - 1) / NGN;
    const int mtiles = (N + 15) / 16;
    for (int item = warp; item < mtiles * NGROUPS; item += nwarps) {
        const int m0 = (item / NGROUPS) * 16, nt0 = (item % NGROUPS) * NGN;
        float acc[NGN][4];
#pragma unroll
        for (int j = 0; j < NGN; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        const float* w0 = W + (size_t)(m0 + g) * KWS + t;
        const float* w1 = w0 + 8 * KWS;
        const float* bcol = in + nt0 * 8 + g;
        // No guard on the cell tiles inside the loop: a tile past the 11th reads the
        // next row of the (finite) buffer and its result is never stored.  A guard
        // would wrap every mma in a warp-reconvergence sequence.
#pragma unroll 1
        for (int k0 = 0; k0 < KP; k0 += 8) {
            float af[4] = {w0[k0], w1[k0], w0[k0 + 4], w1[k0 + 4]};
            unsigned ab[4], as[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ab[i] = tf32_(af[i]);
                if (MODE == 1) as[i] = __float_as_uint(af[i] - __uint_as_float(ab[i]));
            }
            unsigned bb[NGN][2], bs[NGN][2];
#pragma unroll
            for (int j = 0; j < NGN; ++j) {
                const float bf0 = bcol[(k0 + t) * MS + j * 8], bf1 = bcol[(k0 + t + 4) * MS + j * 8];
                bb[j][0] = tf32_(bf0); bb[j][1] = tf32_(bf1);
                if (MODE == 1) {
                    bs[j][0] = __float_as_uint(bf0 - __uint_as_float(bb[j][0]));
                    bs[j][1] = __float_as_uint(bf1 - __uint_as_float(bb[j][1]));
                }
            }
            // term-major: the three products of a tile are NGN instructions apart, so
            // a dependent mma never waits on the one just issued
            if (MODE == 1) {
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], as, bb[j][0], bb[j][1]);
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bs[j][0], bs[j][1]);
            }
#pragma unroll
            for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bb[j][0], bb[j][1]);
        }
        const int ma = m0 + g, mb = m0 + g + 8;
        const float ba = ma < N && bias ? bias[ma] : 0.f, bb = mb < N && bias ? bias[mb] : 0.f;
        float* const ot = out + split * MS;
        // An mma tile spans 16 output rows (ma in [m0,m0+8), mb in [m0+8,m0+16))
        // but ``split`` need only be 8-aligned, so it can fall strictly inside a
        // tile (m0 < split < m0+16). The destination layout is a property of the
        // *row*, not the tile, so ma and mb must each pick their own branch --
        // routing both by m0 silently corrupts whichever half disagrees with it.
        // Mirrors gemm_fma's per-row `n < split` test above.
        auto store = [&](int row, float v0, float v1, float b, int n) {
            if (row >= N) return;
            if (row < split) {
                if (n < NC) out[row * MS + n] = v0 + b;
                if (n + 1 < NC) out[row * MS + n + 1] = v1 + b;
            } else if (row - split < KHID) {
                const int h = row - split;
                if (n < NC) ot[n * PC_T + h] = v0 + b;
                if (n + 1 < NC) ot[(n + 1) * PC_T + h] = v1 + b;
            }
        };
#pragma unroll
        for (int j = 0; j < NGN; ++j) {
            const int n = (nt0 + j) * 8 + 2 * t;
            if (nt0 + j < NTILES_N) {
                store(ma, acc[j][0], acc[j][1], ba, n);
                store(mb, acc[j][2], acc[j][3], bb, n);
            }
        }
    }
}

// Dispatch: float64, or MODE 0, on the FMA path; float with MODE 1/2 on the tensor cores.
template <typename T, int MODE, int KP, int KWS, int NGN, int NTF>
__device__ __forceinline__ void gemm(T* out, const T* in, const T* W, const T* bias, int N,
                                     int warp0 = 0, int nwarps = NWARPS, int split = 1 << 20) {
    if constexpr (sizeof(T) == 8 || MODE == 0) {
        gemm_fma<T, NTF, KP, KWS>(out, in, W, bias, N, warp0, nwarps, split);
    } else {
        gemm_mma<MODE, KP, KWS, NGN>((float*)out, (const float*)in, (const float*)W,
                                     (const float*)bias, N, warp0, nwarps, split);
    }
}

// ---------------------------------------------------------------------------
// pooled[h][c] = mean over the DEG peers of relu(shared[h][c] + per_cell[h][peer])
// pooled[h][c] = mean over the DEG peers of relu(shared[h][c] + per_cell[peer][h]), with
// per_cell cell-major ([NC][PC_T]) so one 16-byte load gathers four h of a peer, and
// relu(s + x) = s + max(x, -s): one load and two flops per (h, peer).  A thread owns a
// cell and a contiguous h range (a multiple of four).  On a KPADDED graph a padded
// port is dropped from the sum and the divisor is the cell's real degree.
template <typename T>
__device__ __forceinline__ void pool(T* __restrict__ pooled, const T* __restrict__ shared_,
                                     const T* __restrict__ per_cell, const Ports<T>& ports) {
    typedef typename Vec4<T>::t V4;
    const int c = threadIdx.x % NC, g = threadIdx.x / NC;
    constexpr int G = NTHREADS / NC;               // h-groups
    constexpr int HPT = RUP((KHID + G - 1) / G, 4);
    if (g >= G) return;
    const int h_begin = g * HPT;
    const T den = KPADDED ? ports.rdeg : T(DEG);
    const T* pp[DEG];
#pragma unroll
    for (int q = 0; q < DEG; ++q) pp[q] = per_cell + PORT(ports, q) * PC_T + h_begin;
    const T* sp = shared_ + h_begin * MS + c;
    T* op = pooled + h_begin * MS + c;
#pragma unroll
    for (int j = 0; j < HPT; j += 4) {
        if (h_begin + j < KHID) {
            T ns[4], a0[4], a1[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ns[i] = h_begin + j + i < KHID ? -sp[(j + i) * MS] : T(0);
                a0[i] = a1[i] = T(0);
            }
#pragma unroll
            for (int q = 0; q < DEG; ++q) {
                if (KPADDED && !((ports.valid >> q) & 1ull)) continue;
                const V4 x = *reinterpret_cast<const V4*>(pp[q] + j);
                T v0 = x.x > ns[0] ? x.x : ns[0], v1 = x.y > ns[1] ? x.y : ns[1];
                T v2 = x.z > ns[2] ? x.z : ns[2], v3 = x.w > ns[3] ? x.w : ns[3];
                if (q & 1) { a1[0] += v0; a1[1] += v1; a1[2] += v2; a1[3] += v3; }
                else       { a0[0] += v0; a0[1] += v1; a0[2] += v2; a0[3] += v3; }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (h_begin + j + i < KHID) op[(j + i) * MS] = (a0[i] + a1[i]) / den - ns[i];
        }
    }
}

// h[f][c] <- GRU(gi + gc, gh, h) elementwise, in place; gates rows [r | z | n] of width F.
// ``gc`` holds this thread's per-cycle constant for its items (nullptr: none).
// ``save``: where to write the gates r | z | n, compact [3F][NC] (nullptr: nowhere).
template <typename T, int F, int ITEMS>
__device__ __forceinline__ void gru(T* __restrict__ h, const T* __restrict__ gi,
                                    const T* __restrict__ gh, const T (*gc)[3],
                                    T* __restrict__ save = nullptr) {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int item = threadIdx.x + i * NTHREADS;
        if (item < F * NC) {
            const int c = item % NC, f = item / NC;
            T ir = gi[f * MS + c], iz = gi[(F + f) * MS + c], in_ = gi[(2 * F + f) * MS + c];
            if (gc) { ir += gc[i][0]; iz += gc[i][1]; in_ += gc[i][2]; }
            const T r = sigmoid_(ir + gh[f * MS + c]);
            const T z = sigmoid_(iz + gh[(F + f) * MS + c]);
            const T n = tanh_(in_ + r * gh[(2 * F + f) * MS + c]);
            const T hx = h[f * MS + c];
            h[f * MS + c] = n + (hx - n) * z;
            if (save) { save[f * NCP + c] = r; save[(F + f) * NCP + c] = z; save[(2 * F + f) * NCP + c] = n; }
        }
    }
    // the record rows' pad columns must be finite: they come back as operand columns
    if (save)
        for (int i = threadIdx.x; i < 3 * F * (NCP - NC); i += NTHREADS)
            save[(i / (NCP - NC)) * NCP + NC + i % (NCP - NC)] = T(0);
}

// this thread's slice of the per-cycle constant, read off its [3F][MS] matrix
template <typename T, int F, int ITEMS>
__device__ __forceinline__ void take_gc(T (&gc)[ITEMS][3], const T* __restrict__ m) {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int item = threadIdx.x + i * NTHREADS;
        if (item < F * NC) {
            const int c = item % NC, f = item / NC;
            gc[i][0] = m[f * MS + c]; gc[i][1] = m[(F + f) * MS + c]; gc[i][2] = m[(2 * F + f) * MS + c];
        }
    }
}

// LayerNorm over the F features of each cell, in place, in one phase: LPC lanes
// per cell each own the features f = q + LPC i, the statistics are reduced with
// shuffles, and every lane normalises what it holds.
#define LPC (NC * 2 <= NTHREADS ? 2 : 1)
static_assert(NC * LPC <= NTHREADS, "LayerNorm lanes");
template <typename T, int F>
__device__ __forceinline__ void layernorm(T* __restrict__ h, const T* __restrict__ gamma,
                                          const T* __restrict__ beta) {
    constexpr int PER = (F + LPC - 1) / LPC;
    const int c = threadIdx.x / LPC, q = threadIdx.x % LPC;
    const bool live = c < NC;             // every lane joins the shuffles
    T v[PER];
    T sum = T(0);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        v[i] = (live && f < F) ? h[f * MS + c] : T(0);
        sum += v[i];
    }
#pragma unroll
    for (int m = 1; m < LPC; m *= 2) sum += __shfl_xor_sync(0xffffffffu, sum, m);
    const T mean = sum / T(F);
    T var = T(0);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        const T d = f < F ? v[i] - mean : T(0);
        var += d * d;
    }
#pragma unroll
    for (int m = 1; m < LPC; m *= 2) var += __shfl_xor_sync(0xffffffffu, var, m);
    const T rstd = rsqrt_(var / T(F) + T(1e-5));
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        if (live && f < F) h[f * MS + c] = (v[i] - mean) * rstd * gamma[f] + beta[f];
    }
}

// (NC, W) row-major global <-> [WP][MS] feature-major on chip (zero padded).
template <typename T, int W, int WP>
__device__ __forceinline__ void load_T(T* __restrict__ dst, const T* __restrict__ src) {
    for (int i = threadIdx.x; i < WP * MS; i += NTHREADS) {
        const int f = i / MS, c = i % MS;
        dst[i] = (f < W && c < NC) ? src[c * W + f] : T(0);
    }
}

template <typename T, int W>
__device__ __forceinline__ void store_T(T* __restrict__ dst, const T* __restrict__ src) {
    for (int i = threadIdx.x; i < NC * W; i += NTHREADS) {
        const int c = i / W, f = i % W;
        dst[i] = src[f * MS + c];
    }
}

template <typename T>
__device__ __forceinline__ void copy_words(T* __restrict__ dst, const T* __restrict__ src, int n) {
    // n a multiple of 4, both 16B aligned; global -> shared (float) through cp.async
    typedef typename Vec4<T>::t V4;
    for (int i = threadIdx.x * 4; i < n; i += NTHREADS * 4) {
        if (sizeof(T) == 4) cp_async16(dst + i, src + i);
        else *reinterpret_cast<V4*>(dst + i) = *reinterpret_cast<const V4*>(src + i);
    }
    if (sizeof(T) == 4) cp_async_wait();
}

// Global -> shared copies go through cp.async (16 bytes, no register staging, every
// copy of a thread in flight at once); the thread waits for its own copies at the end,
// and the barrier that follows publishes them.  The float64 path (activations in a
// global scratch) copies through registers.

// [F][MS] on chip <-> record rows [F][NCP] in global memory, 16 bytes at a time
// (NCP / 4 chunks per row; the pad columns carry whatever the on-chip row holds).
template <typename T>
__device__ __forceinline__ void store_rows(T* __restrict__ dst, const T* __restrict__ src, int F) {
    typedef typename Vec4<T>::t V4;
    constexpr int CH = NCP / 4;
    for (int i = threadIdx.x; i < F * CH; i += NTHREADS) {
        const int f = i / CH, j = (i % CH) * 4;
        st_stream4<T>(dst + f * NCP + j, *reinterpret_cast<const V4*>(src + f * MS + j));
    }
}
template <typename T>
__device__ __forceinline__ void load_rows(T* __restrict__ dst, const T* __restrict__ src, int F) {
    typedef typename Vec4<T>::t V4;
    constexpr int CH = NCP / 4;
    const unsigned long long pol = policy_evict_first();
    for (int i = threadIdx.x; i < F * CH; i += NTHREADS) {
        const int f = i / CH, j = (i % CH) * 4;
        if (sizeof(T) == 4) cp_async16_stream(dst + f * MS + j, src + f * NCP + j, pol);
        else *reinterpret_cast<V4*>(dst + f * MS + j) = *reinterpret_cast<const V4*>(src + f * NCP + j);
    }
    if (sizeof(T) == 4) cp_async_wait();
}
// a contiguous block (a multiple of four elements, 16-byte aligned on both sides)
template <typename T>
__device__ __forceinline__ void copy_block(T* __restrict__ dst, const T* __restrict__ src, int n) {
    typedef typename Vec4<T>::t V4;
    for (int i = threadIdx.x * 4; i < n; i += NTHREADS * 4)
        st_stream4<T>(dst + i, *reinterpret_cast<const V4*>(src + i));
}
// global -> shared block
template <typename T>
__device__ __forceinline__ void load_block(T* __restrict__ dst, const T* __restrict__ src, int n) {
    typedef typename Vec4<T>::t V4;
    const unsigned long long pol = policy_evict_first();
    for (int i = threadIdx.x * 4; i < n; i += NTHREADS * 4) {
        if (sizeof(T) == 4) cp_async16_stream(dst + i, src + i, pol);
        else *reinterpret_cast<V4*>(dst + i) = *reinterpret_cast<const V4*>(src + i);
    }
    if (sizeof(T) == 4) cp_async_wait();
}
// feature-major [F][MS] on chip -> cell-major [NC][PC_T] on chip
template <typename T>
__device__ __forceinline__ void to_cell_major(T* __restrict__ dst, const T* __restrict__ src, int F) {
    for (int i = threadIdx.x; i < F * NC; i += NTHREADS) dst[(i % NC) * PC_T + i / NC] = src[(i / NC) * MS + i % NC];
}
template <typename T>
__device__ __forceinline__ void fill_row(T* __restrict__ row, T value) {
    for (int c = threadIdx.x; c < MS; c += NTHREADS) row[c] = c < NC ? value : T(0);
}
template <typename T>
__device__ __forceinline__ void zero_rows(T* __restrict__ dst, int F) {
    for (int i = threadIdx.x; i < F * MS; i += NTHREADS) dst[i] = T(0);
}

// ---------------------------------------------------------------------------
// The backward's GEMMs.
//
// (1) "AT": out[m][c] (+)= sum_k W[k][m] in[k][c]: the transposed weight as the
//     row operand -- a0 = W[(k0+t) KWS + m0+g], conflict-free for KWS == 4 mod 8.
// (2) "dW": Cacc[m][n] += sum_c d[m][c] X[n][c] with the cells as the contraction:
//     the weight gradient of one board-round, accumulated in a global (L2-resident)
//     per-block buffer laid out like the packed weight (row stride CS).  With a row
//     of ones in X the bias gradient falls out in the padding column.
// BCM: ``in`` cell-major [NC][PC_T]; OUT_CM: ``out`` written cell-major [NC][PC_T] (no ACC then).
template <int MODE, int KP, int KWS, int NGN, bool ACC, bool BCM = false, bool OUT_CM = false>
__device__ __forceinline__ void gemm_mma_at(float* __restrict__ out, const float* __restrict__ in,
                                            const float* __restrict__ W, int M, int warp0, int nwarps) {
    static_assert(!(ACC && OUT_CM), "accumulate into a cell-major output is not needed");
    const int warp = (int)threadIdx.x / 32 - warp0;
    if (warp < 0 || warp >= nwarps) return;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    constexpr int NGROUPS = (NTILES_N + NGN - 1) / NGN;
    const int mtiles = (M + 15) / 16;
    for (int item = warp; item < mtiles * NGROUPS; item += nwarps) {
        const int m0 = (item / NGROUPS) * 16, nt0 = (item % NGROUPS) * NGN;
        float acc[NGN][4];
#pragma unroll
        for (int j = 0; j < NGN; ++j) {
            const int n = (nt0 + j) * 8 + 2 * t;
            if (ACC && nt0 + j < NTILES_N) {
                acc[j][0] = out[(m0 + g) * MS + n]; acc[j][1] = out[(m0 + g) * MS + n + 1];
                acc[j][2] = out[(m0 + g + 8) * MS + n]; acc[j][3] = out[(m0 + g + 8) * MS + n + 1];
            } else { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        }
        const float* wa = W + m0 + g + t * KWS;          // + k0 * KWS
        const float* bcol = BCM ? in + (nt0 * 8 + g) * PC_T + t : in + nt0 * 8 + g + t * MS;
#pragma unroll 1
        for (int k0 = 0; k0 < KP; k0 += 8) {
            float af[4] = {wa[k0 * KWS], wa[k0 * KWS + 8], wa[(k0 + 4) * KWS], wa[(k0 + 4) * KWS + 8]};
            unsigned ab[4], as[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ab[i] = tf32_(af[i]);
                if (MODE == 1) as[i] = __float_as_uint(af[i] - __uint_as_float(ab[i]));
            }
            unsigned bb[NGN][2], bs[NGN][2];
#pragma unroll
            for (int j = 0; j < NGN; ++j) {
                const float bf0 = BCM ? bcol[j * 8 * PC_T + k0] : bcol[k0 * MS + j * 8];
                const float bf1 = BCM ? bcol[j * 8 * PC_T + k0 + 4] : bcol[(k0 + 4) * MS + j * 8];
                bb[j][0] = tf32_(bf0); bb[j][1] = tf32_(bf1);
                if (MODE == 1) {
                    bs[j][0] = __float_as_uint(bf0 - __uint_as_float(bb[j][0]));
                    bs[j][1] = __float_as_uint(bf1 - __uint_as_float(bb[j][1]));
                }
            }
            if (MODE == 1) {
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], as, bb[j][0], bb[j][1]);
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bs[j][0], bs[j][1]);
            }
#pragma unroll
            for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bb[j][0], bb[j][1]);
        }
        const int ma = m0 + g, mb = m0 + g + 8;
#pragma unroll
        for (int j = 0; j < NGN; ++j) {
            const int n = (nt0 + j) * 8 + 2 * t;
            if (nt0 + j < NTILES_N) {
                if (OUT_CM) {
                    if (ma < M) {
                        if (n < NC) out[n * PC_T + ma] = acc[j][0];
                        if (n + 1 < NC) out[(n + 1) * PC_T + ma] = acc[j][1];
                    }
                    if (mb < M) {
                        if (n < NC) out[n * PC_T + mb] = acc[j][2];
                        if (n + 1 < NC) out[(n + 1) * PC_T + mb] = acc[j][3];
                    }
                } else {
                    if (ma < M) {
                        if (n < NC) out[ma * MS + n] = acc[j][0];
                        if (n + 1 < NC) out[ma * MS + n + 1] = acc[j][1];
                    }
                    if (mb < M) {
                        if (n < NC) out[mb * MS + n] = acc[j][2];
                        if (n + 1 < NC) out[mb * MS + n + 1] = acc[j][3];
                    }
                }
            }
        }
    }
}

// ACM: ``d`` is cell-major [NC][PC_T] (a0 = d[(k0+t) PC_T + m0+g]) rather than [M][MS].
template <int MODE, int NGN, int CS, bool ACM = false>
__device__ __forceinline__ void gemm_mma_dw(float* __restrict__ Cacc, const float* __restrict__ d,
                                            const float* __restrict__ X, int M, int ntiles,
                                            int warp0, int nwarps) {
    const int warp = (int)threadIdx.x / 32 - warp0;
    if (warp < 0 || warp >= nwarps) return;
    const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int ngroups = (ntiles + NGN - 1) / NGN, mtiles = (M + 15) / 16;
    const unsigned long long pol = policy_evict_last();
    for (int item = warp; item < mtiles * ngroups; item += nwarps) {
        const int m0 = (item / ngroups) * 16, nt0 = (item % ngroups) * NGN;
        // the running weight gradient is fetched now (L2) and added after the loop
        float2 ca[NGN], cb[NGN];
        float acc[NGN][4];
#pragma unroll
        for (int j = 0; j < NGN; ++j) {
            const int n = (nt0 + j) * 8 + 2 * t;
            acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#ifdef KNODW
            ca[j] = cb[j] = make_float2(0.f, 0.f);   // experiment: no accumulator traffic
#else
            if (nt0 + j < ntiles) {
                ca[j] = ld_hot2(Cacc + (size_t)(m0 + g) * CS + n, pol);
                cb[j] = ld_hot2(Cacc + (size_t)(m0 + g + 8) * CS + n, pol);
            } else { ca[j] = cb[j] = make_float2(0.f, 0.f); }
#endif
        }
        const float* da = ACM ? d + m0 + g + t * PC_T : d + (m0 + g) * MS + t;
        const float* xb = X + (nt0 * 8 + g) * MS + t;
#pragma unroll 1
        for (int k0 = 0; k0 < MS; k0 += 8) {
            float af[4];
            if (ACM) { af[0] = da[k0 * PC_T]; af[1] = da[k0 * PC_T + 8]; af[2] = da[(k0 + 4) * PC_T]; af[3] = da[(k0 + 4) * PC_T + 8]; }
            else { af[0] = da[k0]; af[1] = da[k0 + 8 * MS]; af[2] = da[k0 + 4]; af[3] = da[k0 + 4 + 8 * MS]; }
            // the contraction runs over the cells: pad columns (>= NC) carry whatever the
            // buffers hold, so they are masked out of the row operand here
            if (k0 + t >= NC) { af[0] = 0.f; af[1] = 0.f; }
            if (k0 + t + 4 >= NC) { af[2] = 0.f; af[3] = 0.f; }
            unsigned ab[4], as[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ab[i] = tf32_(af[i]);
                if (MODE == 1) as[i] = __float_as_uint(af[i] - __uint_as_float(ab[i]));
            }
            unsigned bb[NGN][2], bs[NGN][2];
#pragma unroll
            for (int j = 0; j < NGN; ++j) {
                const float bf0 = xb[j * 8 * MS + k0], bf1 = xb[j * 8 * MS + k0 + 4];
                bb[j][0] = tf32_(bf0); bb[j][1] = tf32_(bf1);
                if (MODE == 1) {
                    bs[j][0] = __float_as_uint(bf0 - __uint_as_float(bb[j][0]));
                    bs[j][1] = __float_as_uint(bf1 - __uint_as_float(bb[j][1]));
                }
            }
            if (MODE == 1) {
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], as, bb[j][0], bb[j][1]);
#pragma unroll
                for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bs[j][0], bs[j][1]);
            }
#pragma unroll
            for (int j = 0; j < NGN; ++j) mma_tf32(acc[j], ab, bb[j][0], bb[j][1]);
        }
#pragma unroll
        for (int j = 0; j < NGN; ++j) {
            const int n = (nt0 + j) * 8 + 2 * t;
#ifdef KNODW
            if (acc[j][0] == 12345.678f)   // experiment: keep the products, skip the stores
#else
            if (nt0 + j < ntiles)
#endif
            {
                if (m0 + g < M)
                    st_hot2(Cacc + (size_t)(m0 + g) * CS + n, make_float2(acc[j][0] + ca[j].x, acc[j][1] + ca[j].y), pol);
                if (m0 + g + 8 < M)
                    st_hot2(Cacc + (size_t)(m0 + g + 8) * CS + n, make_float2(acc[j][2] + cb[j].x, acc[j][3] + cb[j].y), pol);
            }
        }
    }
}

// The same two products on plain FMAs (float64, and MODE 0): one thread per output.
template <typename T, int KP, int KWS, bool ACC, bool BCM, bool OUT_CM>
__device__ __forceinline__ void gemm_fma_at(T* __restrict__ out, const T* __restrict__ in,
                                            const T* __restrict__ W, int M, int warp0, int nwarps) {
    const int tid = threadIdx.x - warp0 * 32, nthreads = nwarps * 32;
    if (tid < 0 || tid >= nthreads) return;
    for (int idx = tid; idx < M * NC; idx += nthreads) {
        const int m = idx / NC, c = idx % NC;
        T acc = ACC ? out[m * MS + c] : T(0);
        for (int k = 0; k < KP; ++k) acc += W[(size_t)k * KWS + m] * (BCM ? in[c * PC_T + k] : in[k * MS + c]);
        if (OUT_CM) out[c * PC_T + m] = acc; else out[m * MS + c] = acc;
    }
}
template <typename T, int CS, bool ACM>
__device__ __forceinline__ void gemm_fma_dw(T* __restrict__ Cacc, const T* __restrict__ d,
                                            const T* __restrict__ X, int M, int ntiles,
                                            int warp0, int nwarps) {
    const int tid = threadIdx.x - warp0 * 32, nthreads = nwarps * 32;
    if (tid < 0 || tid >= nthreads) return;
    const int Nn = ntiles * 8;
    for (int idx = tid; idx < M * Nn; idx += nthreads) {
        const int m = idx / Nn, n = idx % Nn;
        T acc = T(0);
        for (int c = 0; c < NC; ++c) acc += (ACM ? d[c * PC_T + m] : d[m * MS + c]) * X[n * MS + c];
        Cacc[(size_t)m * CS + n] += acc;
    }
}

template <typename T, int MODE, int KP, int KWS, int NGN, bool ACC, bool BCM = false, bool OUT_CM = false>
__device__ __forceinline__ void gemm_at(T* out, const T* in, const T* W, int M,
                                        int warp0 = 0, int nwarps = NWARPS) {
    if constexpr (sizeof(T) == 8 || MODE == 0) gemm_fma_at<T, KP, KWS, ACC, BCM, OUT_CM>(out, in, W, M, warp0, nwarps);
    else gemm_mma_at<MODE, KP, KWS, NGN, ACC, BCM, OUT_CM>((float*)out, (const float*)in, (const float*)W, M, warp0, nwarps);
}
template <typename T, int MODE, int NGN, int CS, bool ACM = false>
__device__ __forceinline__ void gemm_dw(T* Cacc, const T* d, const T* X, int M, int ntiles,
                                        int warp0 = 0, int nwarps = NWARPS) {
    if constexpr (sizeof(T) == 8 || MODE == 0) gemm_fma_dw<T, CS, ACM>(Cacc, d, X, M, ntiles, warp0, nwarps);
    else gemm_mma_dw<MODE, NGN, CS, ACM>((float*)Cacc, (const float*)d, (const float*)X, M, ntiles, warp0, nwarps);
}

// ---------------------------------------------------------------------------
// The backward's elementwise phases.

// LayerNorm backward, in place on dy: the input h' is recomputed as the GRU output
// n + (h - n) z from the saved gates.  The per-cell products for the gamma and beta
// gradients, dy * xhat and dy, are written to ``lnp`` [2F][MS] (gamma rows first) and
// summed over the cells by a weight-gradient GEMM against a row of ones.
template <typename T, int F>
__device__ __forceinline__ void ln_bwd(T* __restrict__ dy, const T* __restrict__ h,
                                       const T* __restrict__ rzn, const T* __restrict__ gamma,
                                       T* __restrict__ lnp) {
    constexpr int PER = (F + LPC - 1) / LPC;
    const int c = threadIdx.x / LPC, q = threadIdx.x % LPC;
    const bool live = c < NC;
    T x[PER], g[PER];
    T sum = T(0);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        if (live && f < F) {
            const T n = rzn[(2 * F + f) * MS + c], z = rzn[(F + f) * MS + c];
            x[i] = n + (h[f * MS + c] - n) * z;
            g[i] = dy[f * MS + c];
        } else { x[i] = T(0); g[i] = T(0); }
        sum += x[i];
    }
#pragma unroll
    for (int m = 1; m < LPC; m *= 2) sum += __shfl_xor_sync(0xffffffffu, sum, m);
    const T mean = sum / T(F);
    T var = T(0);
#pragma unroll
    for (int i = 0; i < PER; ++i) { const int f = q + LPC * i; const T d = f < F ? x[i] - mean : T(0); var += d * d; }
#pragma unroll
    for (int m = 1; m < LPC; m *= 2) var += __shfl_xor_sync(0xffffffffu, var, m);
    const T rstd = rsqrt_(var / T(F) + T(1e-5));
    T s1 = T(0), s2 = T(0);           // mean(dxhat), mean(dxhat * xhat)
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        if (f < F) {
            x[i] = (x[i] - mean) * rstd;          // xhat
            const T dxh = g[i] * gamma[f];
            s1 += dxh; s2 += dxh * x[i];
        }
    }
#pragma unroll
    for (int m = 1; m < LPC; m *= 2) { s1 += __shfl_xor_sync(0xffffffffu, s1, m); s2 += __shfl_xor_sync(0xffffffffu, s2, m); }
    s1 /= T(F); s2 /= T(F);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int f = q + LPC * i;
        if (live && f < F) {
            const T dxh = g[i] * gamma[f];
            dy[f * MS + c] = rstd * (dxh - s1 - x[i] * s2);
            lnp[f * MS + c] = g[i] * x[i];
            lnp[(F + f) * MS + c] = g[i];
        }
    }
}

// GRU backward, elementwise: dh' (in dh, in place -> dh = dh' z) with the saved h,
// r|z|n (in place -> dgh_r|z|n) and gh_n; dgi written to ``dgi`` [3F][MS]; the
// per-cycle constant's gradient accumulated in ``dgc`` (nullptr: none).
template <typename T, int F, int ITEMS>
__device__ __forceinline__ void gru_bwd(T* __restrict__ dh, const T* __restrict__ h,
                                        T* __restrict__ rzn, const T* __restrict__ ghn,
                                        T* __restrict__ dgi, T (*dgc)[3]) {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int item = threadIdx.x + i * NTHREADS;
        if (item < F * NC) {
            const int c = item % NC, f = item / NC;
            const T r = rzn[f * MS + c], z = rzn[(F + f) * MS + c], n = rzn[(2 * F + f) * MS + c];
            const T hx = h[f * MS + c], dhp = dh[f * MS + c], gn = ghn[f * MS + c];
            const T dn = dhp * (T(1) - z), dz = dhp * (hx - n);
            const T dpre_n = dn * (T(1) - n * n);
            const T dr = dpre_n * gn;
            const T dpre_r = dr * r * (T(1) - r), dpre_z = dz * z * (T(1) - z);
            dh[f * MS + c] = dhp * z;
            dgi[f * MS + c] = dpre_r; dgi[(F + f) * MS + c] = dpre_z; dgi[(2 * F + f) * MS + c] = dpre_n;
            rzn[f * MS + c] = dpre_r; rzn[(F + f) * MS + c] = dpre_z; rzn[(2 * F + f) * MS + c] = dpre_n * r;
            if (dgc) { dgc[i][0] += dpre_r; dgc[i][1] += dpre_z; dgc[i][2] += dpre_n; }
        }
    }
}

// the per-cycle constant's gradient, registers -> [3F][MS]
template <typename T, int F, int ITEMS>
__device__ __forceinline__ void put_gc(T* __restrict__ m, T (&gc)[ITEMS][3]) {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int item = threadIdx.x + i * NTHREADS;
        if (item < F * NC) {
            const int c = item % NC, f = item / NC;
            m[f * MS + c] = gc[i][0]; m[(F + f) * MS + c] = gc[i][1]; m[(2 * F + f) * MS + c] = gc[i][2];
            gc[i][0] = gc[i][1] = gc[i][2] = T(0);
        }
    }
}

// The pool's backward.  pooled[h][c] = mean_p relu(s[h][c] + x[h][peer_p]):
//   dshared[h][c]  = dpooled[h][c] / deg(c) * #{p valid : x[h][peer_p] > -s[h][c]}
//   dper_cell[h][p] = sum_{c : p in peers(c)} [x[h][p] > -s[h][c]] dpooled[h][c] / deg(c)
// The second sum runs over the *inverse* relation, which is p's own peers on a
// symmetric graph (KSYM, sudoku) and the host's inverse table otherwise.  On a
// KPADDED graph the divisor is per cell, so ``pool_bwd_shared`` divides the
// cell-major dpooled block by it in place (its own cell's rows) and the two
// phases then only sum; unpadded, both divide by the constant DEG as they read.
// s and x are feature-major [KHID][MS]; the second phase overwrites x in place.
// s, x and dp are cell-major [NC][PC_T] (one 16-byte load = four h of a peer); a thread
// owns a cell and a contiguous h range.  dshared is written feature-major [KHID][MS];
// dper_cell is written cell-major *in place over x* (each thread's own row: run after
// pool_bwd_shared, which gathers x).
template <typename T>
__device__ __forceinline__ void pool_bwd_shared(T* __restrict__ ds, const T* __restrict__ s,
                                                const T* __restrict__ x, T* __restrict__ dp,
                                                const Ports<T>& ports) {
    typedef typename Vec4<T>::t V4;
    const int c = threadIdx.x % NC, g = threadIdx.x / NC;
    constexpr int G = NTHREADS / NC;
    constexpr int HPT = RUP((KHID + G - 1) / G, 4);
    if (g >= G) return;
    const int h_begin = g * HPT;
    int off[DEG];
#pragma unroll
    for (int q = 0; q < DEG; ++q) off[q] = PORT(ports, q) * PC_T + h_begin;
#pragma unroll
    for (int j = 0; j < HPT; j += 4) {
        if (h_begin + j < KHID) {
            const V4 sv = *reinterpret_cast<const V4*>(s + c * PC_T + h_begin + j);
            V4* dpp = reinterpret_cast<V4*>(dp + c * PC_T + h_begin + j);
            const V4 dv = *dpp;
            const T ns[4] = {-sv.x, -sv.y, -sv.z, -sv.w};
            int cnt[4] = {0, 0, 0, 0};
#pragma unroll
            for (int q = 0; q < DEG; ++q) {
                if (KPADDED && !((ports.valid >> q) & 1ull)) continue;
                const V4 xv = *reinterpret_cast<const V4*>(x + off[q] + j);
                cnt[0] += xv.x > ns[0]; cnt[1] += xv.y > ns[1]; cnt[2] += xv.z > ns[2]; cnt[3] += xv.w > ns[3];
            }
            T dvv[4] = {dv.x, dv.y, dv.z, dv.w};
            if (KPADDED) {
#pragma unroll
                for (int i = 0; i < 4; ++i) dvv[i] /= ports.rdeg;
                *dpp = make_vec4<T>(dvv[0], dvv[1], dvv[2], dvv[3]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (h_begin + j + i < KHID)
                    ds[(h_begin + j + i) * MS + c] = KPADDED ? dvv[i] * T(cnt[i])
                                                             : dvv[i] * T(cnt[i]) / T(DEG);
        }
    }
}
template <typename T>
__device__ __forceinline__ void pool_bwd_pcell(T* __restrict__ x, const T* __restrict__ s,
                                               const T* __restrict__ dp, const Ports<T>& ports) {
    typedef typename Vec4<T>::t V4;
    const int p = threadIdx.x % NC, g = threadIdx.x / NC;
    constexpr int G = NTHREADS / NC;
    constexpr int HPT = RUP((KHID + G - 1) / G, 4);
    if (g >= G) return;
    const int h_begin = g * HPT;
    int off[DEG];
#pragma unroll
    for (int q = 0; q < DEG; ++q) off[q] = IPORT(ports, q) * PC_T + h_begin;
    const T* sb = s;
    const T* db = dp;
#pragma unroll
    for (int j = 0; j < HPT; j += 4) {
        if (h_begin + j < KHID) {
            V4* own = reinterpret_cast<V4*>(x + p * PC_T + h_begin + j);
            const V4 xv = *own;
            const T xp[4] = {xv.x, xv.y, xv.z, xv.w};
            T a0[4] = {T(0), T(0), T(0), T(0)}, a1[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
            for (int q = 0; q < DEG; ++q) {
                if (KPADDED && !((ports.ivalid >> q) & 1ull)) continue;
                const V4 sv = *reinterpret_cast<const V4*>(sb + off[q] + j);
                const V4 dv = *reinterpret_cast<const V4*>(db + off[q] + j);
                const T svv[4] = {sv.x, sv.y, sv.z, sv.w}, dvv[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const T v = xp[i] > -svv[i] ? dvv[i] : T(0);
                    if (q & 1) a1[i] += v; else a0[i] += v;
                }
            }
            V4 o;
            if (KPADDED) {   // dpooled is already divided by the emitting cell's degree
                o.x = a0[0] + a1[0]; o.y = a0[1] + a1[1];
                o.z = a0[2] + a1[2]; o.w = a0[3] + a1[3];
            } else {
                o.x = (a0[0] + a1[0]) / T(DEG); o.y = (a0[1] + a1[1]) / T(DEG);
                o.z = (a0[2] + a1[2]) / T(DEG); o.w = (a0[3] + a1[3]) / T(DEG);
            }
            *own = o;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase timing (KTIMING builds): block 0's thread 0 accumulates the cycles
// between consecutive barriers into g_timing[phase], read back by ``timing()``.
#define NPHASES 16
#define NSAMPLES 16384
__device__ long long g_timing[NSAMPLES];     // (phase << 40) | cycles, block 0's samples
__device__ int g_nsamples;
#ifdef KTIMING
#define PHASE(i) do { if (blockIdx.x == 0 && threadIdx.x == 0) { long long now_ = clock64(); \
        int at_ = g_nsamples; if (at_ < NSAMPLES) { g_timing[at_] = ((long long)(i) << 40) | (now_ - tlast_); g_nsamples = at_ + 1; } \
        tlast_ = now_; } } while (0)
#define PHASE_BEGIN() long long tlast_ = clock64(); (void)tlast_
#else
#define PHASE(i) do { } while (0)
#define PHASE_BEGIN() do { } while (0)
#endif


// ---------------------------------------------------------------------------
// The host side of the wiring: the four graph tensors, checked once per call
// and handed to the kernels as a Graph<T> of raw pointers.  ``inverse``,
// ``masks`` and ``rdeg`` are always passed; an unpadded symmetric build never
// dereferences them.
struct Wiring {
    const int* peers;
    const int* inverse;
    const unsigned long long* masks;
    const void* rdeg;
};

static inline Wiring wiring_of(torch::Tensor& peers, torch::Tensor& inverse,
                               torch::Tensor& masks, torch::Tensor& rdeg) {
    TORCH_CHECK(peers.dtype() == torch::kInt32 && peers.is_contiguous(), "peers int32");
    TORCH_CHECK(inverse.dtype() == torch::kInt32 && inverse.is_contiguous(), "inverse int32");
    TORCH_CHECK(masks.dtype() == torch::kInt64 && masks.is_contiguous(), "masks int64");
    TORCH_CHECK(peers.size(0) == NC && peers.size(1) == DEG, "peers (", NC, ", ", DEG, ")");
    TORCH_CHECK(inverse.sizes() == peers.sizes(), "the inverse table has the peers' shape");
    TORCH_CHECK(masks.size(0) == NC && masks.size(1) == 2, "masks (", NC, ", 2)");
    TORCH_CHECK(rdeg.numel() == NC && rdeg.is_contiguous(), "rdeg (", NC, ", )");
    Wiring w;
    w.peers = peers.data_ptr<int>();
    w.inverse = inverse.data_ptr<int>();
    w.masks = reinterpret_cast<const unsigned long long*>(masks.data_ptr<int64_t>());
    w.rdeg = rdeg.data_ptr();
    return w;
}

template <typename T>
static Graph<T> graph_of(const Wiring& w) {
    Graph<T> g;
    g.peers = w.peers;
    g.inverse = w.inverse;
    g.masks = w.masks;
    g.rdeg = reinterpret_cast<const T*>(w.rdeg);
    return g;
}

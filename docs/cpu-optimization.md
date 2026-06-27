# CPU optimization and the path to MLAS parity

This is the durable record of the voice-detect.cpp CPU optimization campaign: where
the engine finally stands against the best-in-class references, every technique that
was applied to get there, the dead-ends that should not be re-chased, and a concrete
(not-yet-built) plan for reaching true MLAS-class parity on the conv-bound encoders.

Discipline throughout: `optimizing-ggml-inference-speed` plus the depth-anything.cpp
recipe. Parity-first - profile host-side work before tuning kernels, change one thing
at a time, re-run the parity gate after every change (speaker encoders hold
encoder-out + embedding cosine = 1.000000; wav2vec2 heads hold their max|d| gate and
dominant labels), and keep only end-to-end wins. Numbers below were measured on an
AMD Ryzen 9 9950X3D 16-core (Zen5) on `tests/fixtures/clip_a.wav` (3 s, 16 kHz mono);
absolute ms are machine-specific, the ratios are the signal.

## 1. Final CPU standing

Ratio = reference_ms / ggml_ms at 8 threads (matched thread budget on both sides).
`> 1` means ggml wins. References: onnxruntime / MLAS for the ONNX speaker encoders
(WeSpeaker, ERes2Net, CAM++, ECAPA), torch / MKL for the wav2vec2 analyze heads.

| Model | encoder family | ggml conv path | ratio @ 8t (ref/ggml) | verdict |
| ----- | -------------- | -------------- | --------------------- | ------- |
| WeSpeaker ResNet34 | deep-channel 2D conv (ResNet) | blocked direct-conv microkernel | **0.785** | closest conv encoder to MLAS |
| ERes2Net | deep-channel 2D conv | blocked direct-conv microkernel | **~0.58** | behind |
| CAM++ | 1x1-dominated dense + FCM | mul_mat (1x1) + Winograd (FCM 3x3) | **~0.72** | behind |
| ECAPA-TDNN | 1D TDNN | im2col + sgemm | **~0.66-0.72** | behind |
| wav2vec2 emotion | raw-waveform transformer | GEMM-bound (flash attn) | **~0.74** | behind |
| wav2vec2 age/gender | raw-waveform transformer | GEMM-bound (flash attn) | **> 1 (ggml WINS)** | the one outright win |

Read this honestly: **no conv-bound model reaches MLAS parity.** WeSpeaker is the
closest: it now runs the whole backbone as a single blocked island (Phase A), which
is ~19% faster @8t than the per-conv blocked form the 0.785 was measured on, so its
live @8t ratio is ~0.94 (still short of parity). The single outright CPU win
is wav2vec2 age/gender (a large GEMM-bound transformer where ggml's tinyBLAS plus the
cached positional-conv weight and fused flash attention put it at or over torch/MKL).
The campaign narrowed every gap substantially but did not, on CPU, beat MLAS on a
convolutional speaker encoder.

## 2. Techniques applied (the journey)

Brief, in roughly the order they landed. Each was re-gated for parity before it was
kept.

- **FFT FBank front end.** The original front end was a naive O(T*nfft^2) DFT eating
  ~400 ms/clip (66-82% of every speaker-encoder forward). Replaced with a cached
  radix-2 Cooley-Tukey FFT power spectrum (`src/fbank.cpp`): ~400 ms -> ~3 ms,
  bit-identical embeddings. This was the single biggest win and it unmasked the
  encoder graph as the next bottleneck.
- **Persistent gallocr + fused graph.** Reuse a process-lived `ggml_gallocr` (no
  per-call realloc); the per-call `ggml_context` is metadata-only (`no_alloc=true`).
- **1x1 conv -> `ggml_mul_mat`.** Stride-1 1x1 convs skip im2col entirely and route
  straight to a matmul (bit-identical f32 reduction). Headline lever for the
  1x1-heavy CAM++ backbone and ERes2Net.
- **Layout / context restructures.** Keep the dense backbones channel-major so every
  1x1 conv is one `mul_mat` with no transpose churn; eliminate redundant `ggml_cont`
  copies in the wav2vec2 conv-encoder LayerNorm path; factor the CAM++ context from
  O(T'^2) to O(T' . n_seg).
- **BatchNorm fold.** Fold inference-mode BN affine (`scale = gamma/sqrt(var+eps)`,
  `shift = beta - mean*scale`) to host constants once per loaded model, collapsing a
  multi-op live chain to a single mul + add per BN (algebraic, parity preserved).
- **wav2vec2 head wins.** Cache the positional-conv weight-norm reconstruction
  (`w = g * v / ||v||`) - a pure function of the static weights that was being rebuilt
  on the host every forward (19.7 ms emotion / 112.2 ms age-gender per call,
  bit-exact once cached). Fuse attention via `ggml_flash_attn_ext` (F32 k/v,
  `GGML_PREC_F32`): latency win and tighter parity than the manual QK^T/softmax/xV.
- **The full Winograd ladder.** F(2x2,3x3) custom conv that beat tinyBLAS sgemm on
  the large 3x3 stride-1 convs, then F-blocked, F4, transform-vec, and FCM
  (filter-constant-matrix) variants - the transformed filter cached once per shape,
  tiles split across ggml threads. Default for the FCM 3x3 stride-1 convs.
- **AVX-512 Winograd microkernel.** A hand-vectorized AVX-512 microkernel for the
  Winograd transform/GEMM stage, dispatched at runtime via CPUID so a single binary
  stays correct on AVX2-only hosts.
- **Operand-order tuning (weight-as-src0).** Order the matmul operands so the weight
  is `src0`; the contiguous activation becomes `src1` and routes through tinyBLAS.
- **MLAS-class blocked direct-conv microkernel.** The headline of the final pass: an
  nChw16c blocked, AVX-512 register-tiled DIRECT-convolution microkernel sustaining
  ~245-280 GFLOP/s (0.86-0.98x of the MLAS-implied per-op rate, 77-89% of the Zen5
  f32 FMA peak - measured FMA-bound near peak). It is the default 3x3-stride-1 kernel
  for the deep-channel encoders. WeSpeaker now runs it as a single whole-backbone
  blocked island (Phase A below); ERes2Net/CAM++ still use the per-conv form.

## 3. Confirmed dead-ends (do not re-chase)

- **Linking OpenBLAS / MKL as a second GEMM backend.** Net loss on these small
  batch-1 encoder forwards; the launch/threading overhead and the bandwidth-bound
  shapes mean the vendor BLAS does not beat the already-tuned tinyBLAS path here.
- **q8_0 weights "for speed".** A footprint lever only in this engine. The f32 GEMM
  is already near-optimal and the forward is compute-bound at batch=1, so dropping to
  q8 weights does not buy latency (and it moves parity off bit-exact). Keep it for
  memory, not speed.
- **Global AVX-512 (`-DGGML_NATIVE=ON` / `GGML_CPU_ALL_VARIANTS` picking AVX-512
  everywhere).** Net negative on the conv encoders: AVX-512 sustained-clock downclock
  plus the bandwidth-bound im2col conv regresses the dominant 2D-conv path (ERes2Net
  +21.9% at 8t in the global-AVX512 A/B) even while it helps the GEMM-bound wav2vec2
  heads. One global ISA choice cannot satisfy both families. The win has to be a
  targeted per-op microkernel (which is what the blocked direct-conv kernel is), not
  a global compile flag.
- **Fused-tiled conv.** Our Winograd is already L2-resident with no spill, so an
  additional fusion/tiling layer adds bookkeeping without a measurable win.
- **Broad operand-order sweep.** ggml routes both `(weight, act)` and `(act, weight)`
  orders through tinyBLAS when `src1` is contiguous, so a wide sweep of operand orders
  collapses to the same kernel - no further win beyond the one weight-as-src0 choice
  above.
- **Channel-inner blocked microkernel (contiguous-input reduction).** Tested as the
  "fix" for the blocked layout's width-strided input reads: load the 16 contiguous
  nChw16c input channels as one zmm and reduce over (kh,kw,icb). Implemented and
  parity-verified during the campaign, then **removed** - measured 2-3x SLOWER than
  the width-tiled OC-in-lanes kernel (97/172/93/82 vs 277/280/267/245 GFLOP/s; full
  WeSpeaker forward ~157 ms @1t vs ~67 ms). Both shapes share the same FMA-bound floor,
  but reduction-in-lanes needs a horizontal sum per output scalar, which the OC-in-lanes
  kernel avoids entirely (16 finished channels per store, zero hsum). The width-tiled
  kernel is the correct shape; do not re-implement channel-inner. Full numbers in
  Phase A.

## 4. Path to true MLAS parity (documented plan, not yet built)

The original blocked direct-conv kernel was a **per-conv-reorder** form: it reordered
the activation into nChw16c, ran the blocked microkernel, and reordered back, per conv,
re-paying a reorder tax on every layer. **Phase A below has been built and made the
WeSpeaker default**: the whole backbone now runs as ONE blocked island (2 reorders, not
~60). The other phases are NOT built - and, importantly, the headline ~0.85-0.9x-of-MLAS
projection has been **measured and falsified** (the conv microkernel is already
FMA-bound near peak; the residual gap is elsewhere). The corrected findings follow.

### Phase A - whole-backbone blocked layout (BUILT and DEFAULT for WeSpeaker)

Carry nChw16c across the entire ResNet instead of per-conv: reorder once after the
stem, run all stages in the blocked layout, reorder once before pooling. Built in
`src/directconv.cpp` as a set of blocked-layout custom ops (reorder-in/out, blocked
3x3 conv stride 1/2, blocked strided-1x1 downsample, blocked bias/ReLU/add, plus a
fused conv-epilogue bias+ReLU and a fused residual add+ReLU), each unit-parity-checked
against an NCHW reference (`tests/test_blocked.cpp`). WeSpeaker's forward
(`src/encoder_resnet.cpp`) wires the whole 16-block backbone as ONE island with exactly
TWO boundary reorders (vs ~60 in the per-conv in+out framing). It is the **default for
WeSpeaker when the AVX-512 blocked fast path is available at runtime**
(`directconv_blocked_available()`); non-AVX512 hosts keep the proven AVX2 per-conv
directconv (the blocked ops have a zmm fast path + a scalar fallback only, no AVX2
kernel yet). `VD_BLOCKED_BACKBONE` overrides the gate (`0`/`off`, `all`/`on`, or N
leading blocks).

Measured (encoder-only, AMD Ryzen 9 9950X3D, this clip): the whole-backbone blocked
island runs WeSpeaker at **~64 ms @1t / ~8.7 ms @8t** vs the per-conv directconv
~66 ms @1t / ~10.5 ms @8t - **+2.9% @1t, +~19% @8t**, at exact parity (encoder_out +
embedding cosine = 1.000000, max|d| ~1.6e-6; blocked vs NCHW direct conv share the
kh,kw,ic accumulation order, so the layout change is numerically a no-op).

#### The ~0.85-0.9x-of-MLAS projection is FALSIFIED

The original plan projected ~50-58 ms @1t / ~0.85-0.9x of MLAS, on the theory that the
@1t shortfall was the blocked layout's "still-width-strided input reads" and that a
different microkernel would close it. **That theory was tested and refuted.** An
isolated per-shape microkernel benchmark showed the blocked width-tiled kernel
sustaining **~245-280 GFLOP/s @1t** across the four WeSpeaker stages - matching OR
beating the contiguous-width NCHW direct conv at every shape, and sitting at **77-89%
of the Zen5 f32 FMA peak (0.86-0.98x of the MLAS-implied ~285 GFLOP/s rate)**. The conv
microkernel is already FMA-bound near peak; width-striding is not the cost.

So the residual WeSpeaker -> MLAS gap (**~0.653x @1t**) is **NOT a microkernel-shape
problem.** What remains is (a) sub-peak efficiency on the EXISTING width-tiled kernel -
edge/partial-strip overhead at the small stage-4 spatial - and (b) the non-conv tail:
the 2 boundary reorders, the temporal-pooling mean/std, and the embedding GEMM. Closing
it is kernel-edge tuning plus tail work, not a different reduction order.

**Confirmed dead-end - the channel-inner reduction (do NOT re-try).** The obvious "fix"
for width-striding is a channel-inner microkernel that loads the 16 contiguous nChw16c
input channels as one zmm and reduces over (kh,kw,icb) with 16 OC accumulators. It was
implemented and parity-verified during the campaign (cosine = 1.000000), then
**removed**: head to head it is 2-3x SLOWER (**97/172/93/82 GFLOP/s vs 277/280/267/245**
width-tiled; the full WeSpeaker forward regresses to ~157 ms @1t vs ~67 ms). Both shapes
share the SAME FMA-bound floor (`OC*IC*OutSpatial*9/16` width-16 FMAs), but channel-inner
puts the reduction in the zmm LANES, so it needs one horizontal sum per output scalar;
the width-tiled OC-in-lanes kernel emits 16 finished channels per vector store with zero
hsum. Loading input contiguously saves load-port traffic the kernel was never bound on;
the hsum is pure tax. The width-tiled OC-in-lanes kernel is the correct shape and stays
the only blocked kernel - no dead code shipped, this record is the warning.

### Phase B - tune the existing kernel and broaden the island (NOT yet built)

With the microkernel shape settled, the realistic next steps, in order:

- **(a) AVX2-vectorized blocked path (portability).** The blocked island has only a zmm
  fast path + a scalar fallback, so non-AVX512 hosts fall back to the per-conv
  directconv. A hand-vectorized AVX2 (ymm, 8-OC) blocked kernel would let the island
  (and its 2-reorder framing) run on AVX2 hosts too, the way winograd.cpp already
  dispatches its GEMM per ISA.
- **(b) Profile + tune the width-tiled kernel's sub-peak edge cases + the non-conv
  tail.** The @1t residual is sub-peak edge/partial-strip overhead at the small stage-4
  spatial plus the reorder/pooling/embedding tail, not the reduction order. Attributing
  it needs hardware perf counters (FMA-port utilization, stall breakdown), which is
  currently blocked by `perf_event_paranoid` on the dev host - unblock that first.
- **(c) Expand the blocked backbone to ERes2Net + ArcFace (breadth) - AFTER (b).** Only
  WeSpeaker is wired as a blocked island today; ERes2Net and the ArcFace recognizer body
  are the other deep-channel 3x3 ResNets that would benefit. Wire them once the @1t
  residual is understood, so the same tail/edge cost is not re-paid blind.

### Phase C - perpetual parity maintenance

The direct-conv FMA accumulation order is within-fp32-tolerance, not bit-identical to
im2col/sgemm. Every new variant and every retune must re-pass the parity gate
(cosine >= 0.9999) before it ships. This is an ongoing cost, not a one-time check.

### Economics and decision gate

Realistically a few hundred more lines of intrinsics (the AVX2 blocked kernel) plus the
tail/edge tuning - against a now-measured ceiling. Because the conv microkernel is
ALREADY at 0.86-0.98x of the MLAS-implied per-op rate (77-89% of the Zen5 f32 FMA peak),
there is **no headroom to BEAT MLAS on these shapes**. The earlier ~1.1-1.3x-of-MLAS
ceiling is **withdrawn**; the realistic ceiling is **approaching parity (~0.9-1.0x), not
exceeding it**. The win that remains is closing most of the @8t gap (the island already
takes WeSpeaker from ~0.79x to ~0.94x @8t, +~19%) and a smaller @1t gain bounded by the
non-conv tail. **Decision gate: pursue this only if CPU latency is a measured product
bottleneck. Do NOT build it for the GPU/CUDA path** - on GPU the conv graph is already
competitive and this work is pure CPU.

### Per-model applicability

- **Conv-bound ResNets (WeSpeaker, ArcFace, ERes2Net):** Phases A/B help directly -
  this is where the reorder tax and the per-ISA kernel matter.
- **SCRFD-style detectors (sibling face-detect.cpp):** stay on Winograd by design -
  large-spatial feature maps make the FLOP-reduction win, not blocked direct conv.
- **CAM++:** barely moves; it is 1x1-dominated, already a `mul_mat`, so a blocked
  3x3 kernel has little to grip.
- **wav2vec2 (emotion, age/gender):** not applicable - GEMM-bound transformers,
  already at or over parity; their levers were the pos-conv cache and flash attention,
  not conv kernels.

## Cross-reference

The sibling face-detect.cpp engine shares this parity-path narrative (Phases A/B/C);
see `face-detect.cpp/docs/cpu-optimization.md` for the face-specific standing (SCRFD,
ArcFace, SFace, genderage), where directconv is shape-gated to the ArcFace recognizer
body and SCRFD stays on Winograd. Per-pass measured tables and the GPU comparisons
live in `benchmarks/RESULTS.md` and `docs/benchmarks.md`.

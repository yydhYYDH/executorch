# Attention hang: artifact provenance and the falsified workspace lead

Session 2026-09-17 21:2x-21:4x (dsh host rebooted at 17:50, **/tmp was wiped**, so every
`/tmp` artifact the previous session references — `/tmp/skelbuild`, `/tmp/pristine-mnn`,
`/tmp/fastrpc_pd_probe`, `/tmp/runner_*` — is **gone**).

## 1. Falsified: the "attention workspace too small" lead

Handoff §G② suspected Run B's `out1 (workspace) = 4096` was too small for the real compute
(step A's healthy path had 28672). It is not. Both numbers are exactly what
`_attention_workspace_bytes` prescribes, and the whole difference is the **cache length**.

| arm | cache operand (in1) | cache rows | `seq_len = qo_len + max_kv_len` | padded | bytes/slot | slots | workspace |
|---|---|---|---|---|---|---|---|
| step A (healthy, fast path) | in1/in2 = 557056 B | 272 | 1 + 272 = 273 | 288 | 1152+640 = 1792 | 16 | **28672** |
| Run B (real math) | in1/in2 = 32768 B | 16 | 1 + 16 = 17 | 32 | 128+128 = 256 | 16 | **4096** |

- Host formula: `hexagon_ops.py:1082-1094` — `(align128(qo_len*padded*4) + align128(qo_len*padded*2)) * n_slots`,
  `n_slots = q_shape[2] = 16` (one slot per head, the widest worker count the DSP could pick).
  Runtime call site: `hexagon_ops.py:1175-1177`.
- DSP formula: `attention_sync_setup.cc:185-194` (`sync_attention_head_workspace_bytes`), sized from
  `task_rows` and `state->N = seq_current + seq_add`.
- **At run time the kernel needs far less than the host reserved.** For Run B, `can_group_decode`
  (`attention_sync_setup.cc:228-239`) returns 0 because `N_padded*4 = 128 < head_dim*2 = 256`
  (v79 has no HMX-wide-tile path here), so `decode_grouped = 0`, `task_rows = qo_len = 1`,
  `total_tasks = n_heads = 16`, `n_tasks = min(16, g_max_num_workers = 6) = 6`:
  **6 x 256 B = 1536 B needed of the 4096 B reserved.** No undersized scratch, no OOB write.

`params[8]` is consistent with this reading too: `params[8] = seq_blocks*256`, i.e. 512 for the
272-row cache and 256 for the 16-row one (`hexagon_ops.py:1142,1197`).

## 2. The DSP attention path is (semantically) upstream MNN

`git diff --no-index` of `third-party/mnn-htp-ops/src/dsp` against
`/home/yydh/MNN/source/backend/hexagon/htp-ops-lib/src/dsp` (**MNN master b8c533a5**) differs in
exactly 7 files, and every difference is local instrumentation:

| file | our extra content | semantic? |
|---|---|---|
| `attention_hmx.cc`, `attention_hmx_queue.cc`, `attention_sync_setup.cc`, `attention_common.hpp`, `hmx_queue.cc`, `hmx_mgr.cc` | identical | — |
| `attention_push_kv.cc` | `MNN_CLAMP_CACHE_WRITES` experiment (`push_off_inside`, `g_max_k/v_off`), `htp_probe_stage`, WP_TRACE; upstream has `(void)max_kv_len;` where we use it | switch defaults to 0 = off |
| `execute_command.cc` | the HEXP probe ring + `htp_probe_stage` + per-command probe records | passive |
| `attention_entry.cc`, `attention_sync_process.cc`, `attention_private.hpp` | WP_TRACE only | no |
| `worker_pool.cc` | `MNN_FORCE_SERIAL_WORKERS` (0 = off) + `worker_pool_debug_state` | switch off |
| `vtcm_mgr.cc` | `#include flatbuffers` | no |

So `hmx_queue.cc` (futex/sem queue), `attention_hmx.cc` (`attn_hmx_matmul`, DMA descriptors) and
`attention_sync_setup.cc` are byte-identical to upstream MNN master — the prime blocking suspects
the handoff ranked first are upstream code, not ours.

## 3. Artifact provenance: what the device skels actually are

Fetched the device skels and read their **embedded source paths**:

| device path | md5 | size | built from | when |
|---|---|---|---|---|
| `skel/libhex-htp-skel-v79.so` | 0d0e5be1... | 1465080 | `/home/yydh/executorch/backends/hexagon/third-party/mnn-htp-ops/src/dsp/` | 13:27 |
| `wp_skel_clean/` | 0d0e5be1... | 1465080 | same (identical file) | 14:46 |
| `wp_skel_clamp/` | 6be5b0c6... | 1469432 | same tree, clamp experiment on | 15:18 |
| `skel_pristine/` | 6165f4da... | 1462904 | **`/tmp/pristine-mnn/src/dsp/`** | 11:41 |
| `skel/prev_theirs.so` | 12eff45c... | 1469112 | (not fetched) | 12:28 |
| **fresh, current tree** | **394a2b2e...** | 1465112 | this tree, today 21:30 | now |

- `skel/` vs the fresh build: **identical FUNC symbol sets** (1388 each), ~32 B total size delta ⇒
  `skel/` (0d0e5be1) is our tree as of 13:27 and effectively the same code as today's tree.
- **Both Run B attempts (17:05 `zz_runb.log`, 17:56 `stepB_runner.out`) ran on `skel_pristine`**
  (`zzrun/runbody.sh` sets `ADSP_LIBRARY_PATH=$B/skel_pristine` verbatim), i.e. on a skel built from a
  **separate `/tmp/pristine-mnn` copy that the reboot destroyed**. Handoff §G② and RUNB_REPORT
  deviation (a) both flag this; it is now concrete and **the`pristine` tree is unrecoverable**.
- The 17:56 run *did* use the current-tree runner: device `hxwin/executor_runner_p2` = **29e4db17**
  (= `cmake-out-android-arm64-v8a/executor_runner`, 162594504 B, 17:20:18, newer than every host source).
  The 17:05 run used the older `executor_runner` c79760c2 (its log has no `rpc timeout` line at all).
- Build reproducibility: `SKEL_BUILD_DIR` does **not** affect the md5 (built in two different dirs,
  both 394a2b2e). `IDL_DIR=/home/yydh/et-hexagon/backends/hexagon/idl` works (reboot wiped
  `/tmp/mport/gen2`); `htp_ops.idl` md5 2b73ac08 = MNN's copy.

**Consequence: the "attention math hangs" result has never been reproduced on an artifact built from
this tree.** The only verdict that survives is "a hang at `cmd d0 0` on a skel of unknown provenance".

## 4. Next run (artifacts all fixed and verified)

Already on the device: `hxwin/executor_runner_p2` = 29e4db17, `hxwin/runb.pte` 069f2fb8,
`hxwin/runb_in1.bin` 39d4bc5e, `hxwin/runb_in2.bin` 7dea362b, `hxwin/ctl.pte` +
`ctl_in1/ctl_in2.bin` + golden `hxwin/ctl-0.bin` 1e4430e9.

1. `scp cmake-out-skel-derisk/ship/libhex-htp-skel-v79.so` (**394a2b2e**) to a new dir
   `$B/hxskel/` — do not touch other agents' skel dirs.
2. Gate, `ADSP_LIBRARY_PATH=$B/hxskel` (**single dir** — `A:B` is one path prefix and fails with
   `0x80000406`), runner `executor_runner_p2`, model `ctl.pte`:
   want `idle`/`exit d0: ok`, no `0x12`, `EXIT=0`, and `ctl-0.bin` bit-identical to
   `hxwin/ctl-0.bin` (md5 1e4430e9).
3. Run B: same env, `runb.pte`, `runb_in1.bin,runb_in2.bin`.
   - `exit d0: ok` ⇒ the historical hang was an artifact of `/tmp/pristine-mnn`; go straight to the
     numeric comparison against a host reference.
   - still silent after `cmd d0 0` ⇒ first clean reproduction, then instrument (see §5).
   Use a **writable** `--output_file` (`/dev/null` becomes `/dev/null-0.bin` ⇒ `rc=134` even when healthy),
   and keep `timeout 120` in `runbody.sh` (a SIGTERM does not kill a hung runner, but it bounds the wrapper).
   Never retry a hung attempt; check `uptime` before and after.

## 5. If it hangs again: make the hang readable

Every wait the op can sit in, and the cheapest way to bound it (a fault code the host can print beats
a FARF line, because FARF is unreachable on this non-root device — `Permission denied` on every
watcher dir, and `.farf` is not searched in the ADSP dir; `.debugconfig` *is*, which is a separate
lead worth one try):

| site | code | why it can block forever |
|---|---|---|
| `hmx_queue_submit` final wait | `hmx_queue.cc:163-165` `qurt_futex_wait(&job.done, 0)` | waiting on the queue thread |
| queue thread idle wait | `hmx_queue.cc:90` (v79: `MNN_HMX_QUEUE_POLL_COUNT = 0` ⇒ straight to futex) | waiting on submitters |
| `dma_wait_for_idle()` x13 in `attention_hmx.cc` | `dma_utils.h:94-102`, `Q6_R_dmwait()`, no timeout, result ignored | a DMA that never completes blocks the queue thread |
| `HAP_compute_res_hmx_lock` / `hmx_unit_acquire` | `hmx_queue.cc:57-58`, `hmx_mgr.cc:53` (`qurt_mutex_lock`, not recursive) | CRM/HMX ownership, see the 0x12 story |
| `worker_pool_synctoken_wait` | `attention_sync_process.cc:729` | a worker that never returns |

Plan: behind `-DMNN_ATTN_HANG_PROBE=1` (default off, so the shipping skel is unchanged) — a global
cursor + fault latch, bounded polls (`dmpoll()` + `HAP_perf_get_time_us()` deadline) replacing
`dmwait()`, bounded waits in `hmx_queue_submit` and the synctoken wait, and `htp_ops_flash_attn`
returning `0x1000 + fault` (the host prints it) plus stamping the code into `pOut[0..3]` so the
artefact carries it too. One run then names the site instead of costing a reboot per hypothesis.

## 7. Window round 1 (21:35-22:00): instrumented runs, and what the trail rules out

Diagnostic build `-DMNN_ATTN_STAGE_PROBE=1` (skel 1469240 -> 1482040 B) adds: stage breadcrumbs at
every attention phase boundary, `htp_probe_stage` sites in `hmx_queue.cc`, entry/exit and handoff
liveness counters (stages 64-71), a bounded DMA wait (`dmpoll()` + deadline, writes stage 58 on
fault / 59 on success), and a host-side watchdog thread that dumps the probe ring every
`HEXAGON_WATCHDOG_SECONDS` while the invoke is still blocked.

Runs (every one preceded by the zzh_ms272 gate, `ADSP_LIBRARY_PATH` = the new skel only):

| run | artifacts | result |
|---|---|---|
| gate x4 | skel 394a2b2e / 504b6e7b / e463b17f / 912a5439 | `exit d0: ok`, output md5 **1e4430e9** = golden, every time |
| Run B x3 | v1, v2, v3 probe skels | hangs; the watchdog trail is **identical** in all three |

The trail (stable over 4 watchdog dumps 10/20/30/40 s, `hx_prunb3.log`):

```
1,2,3   flash_attn entered, pre/post push_kv
4       pre sync_attention      (worker_slots 6, task_rows 1, total_tasks 16)
32..36  run_tasks entered; pre/post hmx_queue_begin; submits done; pre synctoken wait
39      process_head entered    (head 5, worker 5, qo_len 1)
40      pre causal QK
47,48   attn_hmx_matmul entered (M=1 K=128 N=32); vtcm sized (kp 4, np 1, pair_packs 2)
49      pre compute
     59 dma wait ok  site 4 (post weight DMA), elapsed 1 us      <- no wait ever timed out
60..63  weight desc: src=0x411000 dst=0xfd000000 w=8192 h=1, strides 8192/8192, cache_alloc 2,
        layout 0 (K_BLOCK256), v_ocP 1, kv_head 0
50      post store             (ox 0, oy 0..1)
     64 matmul enters = 1      65 matmul exits = ABSENT
     66 queue idle wakes = 2   67 queue jobs done = 1 (type END)
     68 submit completions = 1 (END)   69 submit sem acquires = 7   70 worker iters = 6
57      submit blocking wait   (spin 2000, read_index 2)
```

What that establishes:

- **Exactly one matmul was ever entered, and none ever returned.** The queue thread's last job
  (read_index 2) is a CALLBACK whose body is that matmul: it was entered, stored its scores, and
  never reached stage 51.
- **No DMA wait timed out and no DMA wait faulted**: the bounded `dmpoll()` loop reports 59 for the
  last successful wait (site 4, 1 us) and 58 never appears. The QK's weight DMA completed. So the
  prime suspect from the handoff's census - `dma_wait_for_idle()` - is **not** what blocks here.
- Stages 64-71 froze at their values above and did not advance across the four dumps, so by the time
  the host looks, the queue thread has stopped cycling entirely.

Remaining hypothesis space (next round bisects it): after stage 50 there is *no* blocking primitive
in the source, so either the DSP PD faulted/took an exception at that point (which on this platform
surfaces as an RPC that never returns, because status-notification registration fails with
`0x0000000e` and no RPC timeout is armed), or the thread is stuck inside the cache flush that
`htp_probe_stage` performs for stage 50. Bisect plan: two builds that skip `attn_store_output_*`
and `attn_hmx_compute_output_tiles` respectively - if either run completes, the culprit is named.

Device note: the phone rebooted **on its own twice** during this window (≈21:45 and ≈21:58), and a
co-tenant `llama-cli` (21:50:24, ending in `dspqueue` timeouts) was on the DSP during the v2 run, so
only the v1 and v3 runs are clean. A reboot clears a wedged session; the v3 artifacts survived it.

## 8. Window round 2 (22:00-22:15): bisect says the wedge is in the DMA engine itself

Build switch `MNN_ATTN_BISECT` (0 normal, 1 skip compute+store, 2 skip store, 3 skip compute) plus
`MNN_ATTN_WEIGHT_CACHEALLOC` (default `DMA_DESC_CACHEALLOC_READONLY`), both compile-time.

**Run B with `MNN_ATTN_BISECT=1`** (skel f1b20cd7, skip the HMX compute and the store, keep both DMAs),
gate first on the same skel (RC=0, `out_pctl4` md5 1e4430e9 = golden):

| stage | value | reading |
|---|---|---|
| 47 / 48 | (1,128,32) (4,1,2) then (1,9,128) (1,4,1) | QK then SV, kp/np/pair_packs as derived for this geometry |
| 51 | **(1,128,0)** | stage 51 exists at all - impossible in the previous three runs |
| 64 / 65 | **20 enters / 19 exits** | matmuls now run and return |
| 72 / 73 | counts of weight/activation `dmstart` | see below |
| **58** | **(2, 1, 500)** | **bounded DMA wait fault: site 2 = post-`dmstart` activation load, status 1 = RUN, 500 ms** |
| 59 | (1,1,0) | the pre-`dmstart` wait at that same call site was idle, 0 us earlier |
| rc | 134 (SIGABRT), **0 watchdog dumps** | the invoke *returned* the fault code instead of hanging |

So with compute+store removed: 19 matmuls complete, then the 20th matmul's **activation DMA starts
(engine goes RUN) and never completes**. No HMX operation ran in this build at all, so the wedge is
not HMX-induced; it is a DMA/hardware-level stall that accumulates over ~20 matmuls.

That also explains the earlier signature, which had no blocking primitive between stages 50 and 51:
`htp_probe_stage` flushes its record with `qurt_mem_cache_clean(INVALIDATE)`, and that call has no
timeout. When the engine is wedged, the *flush of stage 51* is where the thread stops, leaving stage
50 as the last visible record - exactly the trail measured in all three earlier runs. The bounded
`dmpoll()` wait added here is the first instrument that can *report* the wedge instead of joining it.

Falsified this round: stale `next` in the chained descriptors (`attn_prepare_dma_desc_2d` does
`memset(desc, 0, sizeof)` and then links explicitly, so the chain terminator is always 0).

Ready to fire (all built locally, dumps included: stages 74-78 record the failing 2-D descriptor -
src/dst/next/flags/w/h/strides/cache_alloc - and the weight chain's terminator):

| build | flags | md5 | purpose |
|---|---|---|---|
| VA | probe + normal + `cache_alloc=0` on weight DMA | e265f894 | is the cache-allocating weight DMA the thing that wedges the engine |
| VB | probe + bisect=1 + `cache_alloc=0` | 90d2ad32 | does the 20-matmul wedge disappear when nothing allocates cache lines |
| ref | probe + normal (as v3) | 912a5439 | unchanged baseline, still the one on the device |

Numeric grading is now prepared too: `prov/runb_ref.py` reproduces the fixture RNG (q md5
39d4bc5e = the device's `runb_in1.bin` byte for byte) and emits two candidate expectations -
`prov/runb_ref_out.bin` (md5 328b3d13, the eager torch reference, cross-checked against a hand-rolled
causal GQA sdpa to 0.000000) and `prov/runb_ref_cold.bin` (md5 a71cff29, what a cold packed cache
must produce when push_kv inserts only the row at seq_current). A device run that stops hanging can
now be graded against both instead of only being declared alive.

## 9. Debug plan: turn the wedge into a metered, self-terminating experiment

Framing. The object is not "which line of attention math is wrong" (19 matmuls ran and returned) nor
"which attention phase hangs" (the stage trail lied once: the flush inside `htp_probe_stage` is what
stops, because `qurt_mem_cache_clean` has no timeout). The object is: which hardware path puts the
DMA engine into a state it never leaves, and what accumulates until then.

Two enablers, both now in place:

1. **A reproducing build that returns.** `MNN_ATTN_BISECT=1` (f1b20cd7) wedges at the 20th matmul and
   then *returns* the fault code, so a device shot costs ~1 minute and never has to be killed.
2. **A metered workload.** `executor_runner --num_executions=N` runs the same command N times in one
   process, and the liveness counters (stages 64-73) are static (never reset), so a single run reports
   the *cumulative* `dmstart` count at each wedge. That separates "fixed budget per op" from "budget
   per PD lifetime" from "race", which decides the whole subsequent strategy.

Also `MNN_ATTN_FAULT_IS_SOFT=1` keeps a latched fault in the stage ring while returning success, so a
failing execution no longer aborts the runner and the following executions still run.

Variant matrix, one build each, one property differs per shot:

| axis | build | flags | tests |
|---|---|---|---|
| reference repeat | VC 8080ee75 | probe + bisect=1 | the wedging config, now metered: which transfer number, what descriptor |
| A descriptor | VA e265f894 / VB 90d2ad32 | normal / bisect=1, both + `cache_alloc=0` | is the cache-allocating weight DMA the resource that runs out |
| A descriptor | VF (building) | bisect=1 + `src_bypass=1` | source read through the cache path at all (src is host DDR, 0x411000) |
| A descriptor | next | descriptors in static storage instead of the stack; unordered; chain off | descriptor lifetime / ordered-write semantics |
| B concurrency | next | global mutex around every `dmstart`; `wait_after=1` always | several threads driving one engine |
| C memory | next | weight source copied into DSP-heap before the DMA | host-mapped buffer cache/SMMU attribute handshake |
| D budget | VCS (building) | bisect=1 + `FAULT_IS_SOFT=1` | the budget curve across repeated executions |

Read-outs per shot, all host-side: stage 58 (site + engine status + ms), 59 (last good wait, us),
60-63 (successful weight descriptor), 74-78 (the *failing* descriptor, both DMA kinds, plus the weight
chain terminator), 64-73 (counts). Prediction per axis is crisp, so a shot either kills a hypothesis or
promotes it - no run is wasted on "look and see".

Grading a fix needs numerics, and that is prepared too: `prov/runb_ref.py` regenerates the fixture
RNG (q md5 39d4bc5e = the device's runb_in1.bin) and writes both candidate expectations
(`runb_ref_out.bin` 328b3d13 eager torch, cross-checked against hand-rolled causal GQA sdpa to
0.000000; `runb_ref_cold.bin` a71cff29 for a cold packed cache).

Outside oracle still unused: the third-party `attn-hexkl/bin/fa_std_test` harness drives the same
vendored kernel. If it completes, diffing its invocation against ours localises the bug in one step;
if it hangs too, the bug is vendored-kernel x this device (v79 / HAP version), which narrows the axes
to A/B/C above. No host copy exists, so this needs a read-only look on the device and the owner's ok.

## 10. Root cause: the DSP holds no power/DCVS vote during the run (fixed)

`htp_ops_global_backend_setup()` (`src/dsp/commu.cc:111-118`) is

    power_acquire();
    vtcm_manager_setup(); hmx_manager_setup(); hmx_queue_setup();
    power_release();

so the refcount returns to zero, `power_reset()` (`power.cc:124-141`) powers HMX down and drops
the DCVS request that `power_setup` had pinned at max corners with `sleep_disable = TRUE`
(`power.cc:33-59`). The only re-acquire, `HexagonDriver::PowerAcquire` (`hexagon_driver.cpp:553-559`),
has no caller anywhere in the tree. Nothing holds the vote while the op runs, so the SoC is free to
change corners or gate a domain mid-transfer; a DM transaction then has no responder, its status stays
RUN (never ERROR), the PD is wedged for the rest of the device's uptime, and the SoC watchdog
eventually resets the phone - which is the observed reboot pattern.

Evidence, same fixture, same binary, one variable:

| build | power vote | executions | result |
|---|---|---|---|
| VCS / VB / VC (bisect, vote released) | released | 1 each | wedge at the 20th activation DMA (stage 58 site 2, RUN, 500 ms), 3 times in a row |
| VD 8080ee75->c079fd0b (bisect, vote released) | released | 1 | **completed all 32 matmuls** (72=73=32, no stage 58) |
| VD, same binary | released | 3 | **wedged at 19/20 again** |
| VE 5cddc72a (bisect + vote held) | held | 6 | **6/6 completed**: 72 = 73 = 32,64,96,128,160,192; no stage 58 anywhere |

The workload is deterministic, so a time-triggered external event lands on the same matmul index
(19/20) for this fixture and on the 1st for the paged fixture, whose transfers are far larger. One
completion in four runs rules out every deterministic code defect: not the descriptor values (the
failing descriptor is well formed and shape-identical to the 19 that succeeded), not `cache_alloc`
(VB, falsified), not the chain terminator (falsified), not the programming model choice (the paged
entry `htp_ops_flash_attn_pages` wedges too, after 1 matmul), not HMX math (bisect runs no HMX).

Two hygiene fixes went in with the diagnostic builds and matter for every later run:
`attn_hmx_load_activation_dma` and `attn_hmx_start_weight_dma` no longer `dmstart` after a latched
fault (they used to submit into a RUN engine whose descriptor sits on a stack frame the next matmul
reuses), and `htp_ops_flash_attn_pages` now clears the latch like the non-paged entry does.

Builds: VA e265f894, VB 90d2ad32, VC 8080ee75, VCS 5d474afd, VF 342dd9e5, VD c079fd0b,
VE 5cddc72a, VG (normal + hygiene, no vote) and VH (normal + hygiene + vote held) are the
numeric-verification pair.

## 11. Numerics verified, and the second bug was in our fixture, not the kernel

With the power vote held (VI, normal path, no bisect) Run B returns and all 32 matmuls complete.
The host copy-out is layout-blind (`hexagon_backend.cpp:1117-1127`: one size check, then a raw
memcpy), so the DSP bytes are what the caller sees. Reading the runner's output as **fp32**
(the graph output is fp32; 8192 B = 2048 fp32) gives values of the right magnitude (|max| 0.13)
but unordered: per-head correlation against the expected V row was ~0.01.

Matching each head's 128 values against all 128 candidate V vectors (16 rows x 8 kv heads) settled
it: every head is an exact scaled copy (corr = 1.000000) of `V[0, row 0, kv]` with kv = head/2.
So softmax, the GQA grouping, the SV matmul and the output layout are all correct; the token was
simply read from the **first row** of the K/V operand.

Reason: `attention_entry.cc:199-203` pushes with `htp_ops_push_kv(pPastK, pPastV, pK, pV, seq_current,
seq_add, ...)`, and the push contract is MNN's - K/V hold only the **new** tokens (`insert_len =
K->length(1)`, HexagonAttention.cpp:390-408) - so the kernel copies operand rows `[0, seq_add)` into
the packed cache at `seq_current`. Our fixture hands it the whole 16-row K/V tensor, so seq_add=1
copies row 0. The emitter never passes insert_len; the DSP cannot know the difference.

Therefore, with the operand semantics the kernel actually implements (cache = zeros, operand row 0
pushed at row 0, valid rows 0..8), a torch reference reproduces the device output:

    max|d| = 5.71e-05   mean|d| = 6.85e-06   corr = +0.99999995
    2048/2048 elements within 2^-11 (relative max|d| 4.4e-04, fp16 precision)

Two consequences for the handoff:
- the kernel's attention math is verified against an independent torch reference; the earlier
  "garbage" was a fixture-side contract mismatch, not a DSP defect;
- our host must narrow the K/V operand to the new tokens (insert_len) before the push, or bake
  seq_current's row into row 0 of the operand. `runb_ref_cold.bin` (a71cff29) encodes the intent
  "rows 0-7 zero, the row-8 token pushed"; the device only matches it once the operand carries
  that single row.

Also ported while tracking this down (all guarded, defaults unchanged unless noted): the missing
`ATTN_HMX_OUT_LINEAR_FP16 = 2` enum, its two store branches, and the terminal `power_release()`
now defaults to holding the vote (`-DMNN_ATTN_HOLD_POWER_VOTE=0` reproduces the pre-fix hang).
## 12. Multi-token (prefill) verified; the output layout for qo_len>1

Fixture prov/runb_mt_build.py (T = RUNB_T, default 4) corrects the K/V operand contract: the
operand holds exactly the T tokens being inserted at POS..POS+T-1, so seq_add = qo_len = T matches
what the push reads ([0, seq_add)). Both runs return on the fixed skel, no watchdog, no reboot.

| run | operands | result |
|---|---|---|
| T=1 | q 4096 B, K/V 2048 B each (one token) | matches runb_ref_cold.bin (a71cff29): max abs diff 6.1e-05, corr 0.99999994 |
| T=4 | q 16384 B, K/V 8192 B each (four tokens) | row-major read is garbage (corr 0.026); pack-major gives max abs diff 1.221e-04, corr 1.000000 against the causal reference |

For qo_len>1 the DSP writes the pack-major layout

    offset(h, pack, token, i) = (h * (head_dim/64) * qo_total_len + pack * qo_total_len + token) * 64 + i

with a per-head stride of (head_dim/64)*qo_total_len*64 elements = 512 elements (1024 B) at T=4,
so a plain row-major [token][head][dim] read sees nothing of the real output. The non-causal
reference scores only corr 0.79, i.e. the causal mask is applied correctly per query row.
Cross-check: token 0 of the T=4 run is bit-identical to the T=1 output (max abs diff 0.0, corr 1.0),
as it must be, since both attend only rows 0..8.

Host consequence, NOT fixed here: our ET copy-out is layout-blind (hexagon_backend.cpp:1117-1127), so
a delegated attention with qo_len>1 currently hands pack-major bytes to the caller. How MNN @ b8c533a5
lives with it: it never moves the layout inside the DSP. The output tensor is declared NC4HW4 with dims
[b*qo_len, n_heads*head_dim, 1, 1] (HexagonAttention.cpp:370, ShapeAttention.cpp:109-116), the hexagon
device NC4HW4 order IS this pack-major formula (HexagonBackend.cpp:620-630, 710, 778), and the conversion
to natural [token][head][dim] happens only on read-back (deviceToNchwHost, HexagonBackend.cpp:906 and
764-787; an NC4HW4 host tensor goes through nc4hw4DeviceToHost, :904 and 694-719). A grep of the newest
DSP confirms there is no value_c4-driven output store at all: attention_common.hpp never mentions
value_c4, and value_c4 only selects V's input layout. The 08-15 tree's value_c4-aware
sync_attention_store_output (which the earlier report described) is NOT in b8c533a5 - do not port it.

So the fix on our side is the same thing MNN does, host-side: de-interleave an attention output whose
qo_len>1 from pack-major to [token][head][dim] during copy-out. hexagon_backend.cpp needs the op's
qo_len / n_heads / head_dim to do it, and the emitter already bakes exactly those into the op params
(params[0], [3], [5]). qo_len == 1 is unaffected: the formula degenerates to plain [head][dim], which
section 11 verified.

Two latent MNN issues found while reading, worth flagging upstream (not ours):
- 2 <= qo_len <= 8 with seq_current+seq_add > 2048 diverts to sync_attention_process_online_pages
  (attention_sync_process.cc:470-472), which sets rows = gqa_factor, global_q = 0 and writes the
  qo_len==1 layout (:237-241, :288-294) - wrong values and wrong layout for prefill.
- sync_attention_store_output_value (attention_common.hpp:601-607) uses qo_len where the rest of the
  writeback uses qo_total_len; only reachable at qo_len == 2 (:610-611), where segmentation cannot be
  active, so it is latent rather than live today.

## 6. Device window 2026-09-17 21:35-21:39 (exclusive, user-approved): clean reproduction

Artifacts, all verified on the device before the run: skel **394a2b2e** (this tree, deployed to a new
`hxskel/`), runner **29e4db17** (`hxrun/executor_runner_p2`), `ctl.pte` = zzh_ms272 **98adcdce** with
`ctl_in1` bd1b5b82 / `ctl_in2` 7dea362b plus golden `ctl-0.bin` **1e4430e9**, `runb.pte` 069f2fb8 with
`runb_in1` 39d4bc5e / `runb_in2` 7dea362b. Device idle (PJZ110, up 1:47), `runbody.sh` fixed:
`ADSP_LIBRARY_PATH=$B/hxskel` (single dir), `timeout 120`, writable `--output_file`.

1. **Gate PASSED** — `hx_gate.log`: `enter d0` -> `exit d0: ok`, no `0x12`, `EXIT=0`,
   `out_ctl-0.bin` md5 **1e4430e9** = golden. The fresh skel is healthy on the device.
2. **Run B HANGS**, same signature as ever — `hx_runb.log` stops at
   `[hexagon] enter d0: ops=1 act=1052672 first=0 count=1`; no `cmd d0 0` record, no `exit d0: ok`,
   no rc file, no `out_runb-0.bin`.
   Kernel evidence: pid 23381 `executor_runner` -> thread **23387 state=`D`, wchan=`fastrpc_wait_for_completion`**;
   the `timeout` wrapper is stuck in `sigsuspend`. Signals cannot reach a `D`-state thread, which is
   exactly why `timeout 120` / `kill -9` do nothing and why only a reboot used to "clear" it.
   logcat shows the skel opening and `Model loaded in 55.3 ms` and then nothing.

**So: the hang is real, deterministic, and now reproduced on an artifact built from this tree, on an idle
device, with a bit-exact healthy control immediately before. `/tmp/pristine-mnn`, device contention and
the old runner are all eliminated as explanations.** The wedged process holds a user PD but does not block
later runs (a new process gets a fresh PD) — the 17:39 run already proved that.

What is still unknown is *where* it blocks. Next: DSP breadcrumbs (`htp_probe_stage`) at the attention
phase boundaries plus a host-side watchdog thread that dumps the existing HEXP probe ring after N seconds
of a blocked invoke — that turns this hang into a cursor.

## 13. Fix: the non-C4 attention output is now written linearly (verified)

Sections 10-12 describe the outcome; this is the fix that closes the layout half of it. The DSP
writeback is now driven by value_c4, the flag our emitter already sets to 0 for every attention op.
MNN keeps value_c4 = 1 and consumes the packed device layout through NC4HW4 host tensors; a non-C4
consumer (ExecuTorch) gets the natural [token][head][head_dim] order instead.

Plumbing: SyncAttentionTaskState gained a value_c4 field, threaded from htp_ops_flash_attn and
htp_ops_flash_attn_pages through sync_attention / sync_attention_pages / sync_attention_init_common_state
(attention_private.hpp, attention_sync_setup.cc, attention_entry.cc). Two stores became conditional:

- attention_sync_process.cc, decode-group writeback (the qo_len>1 branch, also taken when pages are on):
  value_c4 != 0 writes ((head*(head_dim/64) + pack)*qo_total_len + token)*64 + i as before, while
  value_c4 == 0 writes (token*total_heads + head)*head_dim + pack*64 + i.
- attention_sync_process.cc, sync_attention_run_causal_sv_block, non-paged branch (the process_head path
  taken when gqa == 1 or an explicit mask is present): ATTN_HMX_OUT_LINEAR_FP16 with the token stride
  total_heads*head_dim and head base O + head_id*head_dim, where it used to store packed. Its paged
  branch is deliberately unchanged and still writes packed - marked with a FIXME, because the paged SV
  store has no linear form.
- htp_ops_vision_flash_attention_fp16 passes value_c4 = 1 on purpose: it consumes a packed staging
  buffer, so its bytes are identical to before, which the golden gate confirms.

Evidence (skel 94db1b4e = VR, probes off, power fix on; grading by prov/grade_mt.py):

| run | operands | output md5 | row-major vs the causal reference |
|---|---|---|---|
| gate ctl.pte, before and after | - | 1e4430e9 both = golden | healthy path byte-exact |
| T=1 | K/V = the one new token | 17aed9fe (unchanged) | max abs diff 6.1e-05, corr 1.000000 |
| T=4 | K/V = the 4 new tokens | 24f8ffe8 -> 47db8b84 | max abs diff 1.221e-04, corr 1.000000 (pack-major now 0.026) |
| T=8 | K/V = the 8 new tokens | 73405773 -> 46f95ba9 | max abs diff 1.221e-04, corr 1.000000 (pack-major now 0.036) |
| gqa=1, T=4 | n_kv_heads = n_heads = 16 | f3d94365 | max abs diff 1.221e-04, corr 1.000000 (pack-major now 0.034) |

Every run returned RC=0, no watchdog, no device reboot. The non-causal reference still fails (corr
0.74-0.79), so the per-query causal extent is untouched. Reproduce: prov/runb_mt_build.py (RUNB_T,
RUNB_KV) builds a fixture and both references, grade_mt.py reports whichever layout the buffer holds.

Still open: the paged process_head branch above (no numeric acceptance run yet), and the probe
instrumentation is still compiled into the diagnostic skels - the shipped one (0474554e) has it off.

### 13a. The paged path is closed too

The paged SV kernel (attn_hmx_matmul_pages_sv) writes through four store sites and three helpers, none of
which knew about layouts. They now dispatch on value_c4 through sync_attention_store_output,
sync_attention_zero_output and sync_attention_accumulate_output (attention_common.hpp): value_c4 != 0 keeps
MNN's packed device order - the caller passes the packed head base and block stride, as before - while
value_c4 == 0 calls attn_store_output_linear_fp16 / writes the linear equivalent, with the caller passing
the head base of a plain [token][head][head_dim] tensor (O + head_id*head_dim) and the token stride in
elements (total_heads*head_dim). Both paged callers in sync_attention_process_head were updated that way:
the causal branch inside sync_attention_run_causal_sv_block and the explicit-mask branch. The value_c4 != 0
arms are byte-for-byte what they were, so the vision path is untouched: golden gate 1e4430e9 both before
and after these runs.

Evidence, skel 0474554e (VR2), fixtures built with HEXAGON_ATTN_PAGED=1 (page_count = 1, page_size = 256,
value_c4 = 0, so the host takes htp_ops_flash_attn_pages):

| run | output md5 | row-major vs the causal reference |
|---|---|---|
| gate ctl.pte before and after | 1e4430e9 both | healthy path byte-exact |
| paged T=4, kv_heads=8 (decode-group) | 47db8b84 = the non-paged T=4 md5 | max abs diff 1.221e-04, corr 1.000000 |
| paged T=4, kv_heads=16 (process_head) | f3d94365 = the non-paged gqa=1 md5 | max abs diff 1.221e-04, corr 1.000000 |

Both paged runs returned RC=0 with no watchdog and no reboot, and their bytes matching the non-paged runs
exactly is an independent cross-check that the two paths compute the same thing.

Still open, and deliberately not touched: the online-pages path. It exists for contexts past the fixed
workspace budget (ATTN_FIXED_WORKSPACE_KV = 2048): it streams the paged cache one page at a time with a
running max/sum per query row (online softmax), keeping the workspace at O(page_size), and the DSP entry
hardcodes allow_online_pages = 1, so it applies whenever seq_len > 2048 or mask_stride > 2048. It is a
decode-shaped path, and its two exits show it: with gqa > 1 it runs with rows = gqa_factor and
q_ptr = Q + head_base*head_dim, i.e. only token 0's query is computed, and it exits through
sync_attention_copy_decode_group_output, the qo_len == 1 tail, writing token 0's row-major slots - wrong
values for 2 <= qo_len <= 8, an upstream defect, not a layout choice we can make here. Without grouping
(gqa == 1 or an explicit mask) rows = qo_len and the exit is sync_attention_copy_scaled_packed_output with
the packed head base, so a non-C4 multi-token caller still gets pack-major. Plain qo_len == 1 decode,
which is what the path is for, is correct either way because the packed and linear layouts coincide at one
token. The explicit-mask path is implemented but has no fixture yet, since the emitter always passes
mask_stride = -1.

## 14. The external "Run B numerics are wrong" report is the legacy fixture's operand extent

An external recipe (own build dir, own skel, `hxwin/runb.pte` + `runb_in1/2.bin`, graded against
`prov/runb_ref_cold.npy` / `prov/runb_ref_out.npy`) reports max abs diff 1.8e-01 at corr 0.015 and
2.6e-01 at corr 0.287 and concludes the numerics are wrong. That is the fixture-side contract mismatch of
§11, still reproducible, so this section pins it down instead of restating it.

**Their run is deterministic and is the legacy fixture.** The blob it produced is byte-identical to
`prov/runb_dev_out.bin`, `prov/runb_dev_lin.bin` and `prov/runb_dev_pos8.bin` (all md5 604da444), and the
device `hxwin/runb.pte` (069f2fb8) is byte-identical to `prov/zzb_runb_np.pte`. So the recipe's md5
"expectation" 604da444 is the wrong-insertion result recorded as expected: it grades liveness, not
numerics.

**The two fixtures differ in exactly one field.** Same emitter, same params
`qo_len=1 seq_current=8 seq_add=1 n_heads=16 n_kv_heads=8 head_dim=128 mask_stride=-1 max_kv_len=256
page_count=0 page_size=0 value_c4=0`, same q (`39d4bc5e`, the 4096 B Q operand) and same pos
(`7dea362b`):

| fixture | K operand | V operand |
|---|---|---|
| legacy `zzb_runb_np.pte` = device `runb.pte` | 32768 B = the whole 16-row packed cache | 32768 B |
| `zzb_runb_mt1.pte` | 2048 B = one token (1x8x128 fp16) | 2048 B |

The push contract is MNN's: operand rows `[0, seq_add)` are inserted at `seq_current`. Handing over the
whole cache makes the kernel insert **operand row 0**, the seq-0 row, at slot 8.

**What the kernel computed, by inversion rather than argument.** Rebuilding the fixture RNG and scoring
candidate caches against the blob gives `rows 0..7 zero + K/V row0` at max abs diff 5.7e-05, corr
1.000000, while the two references sit at 0.015 (cold, row 8 inserted) and 0.287 (eager over 16 rows).
Least-squares solving `out[h] = sum_r w_r * V[r, kv(h)]` is exact (residual ~2e-05) with a single
non-zero weight per head, on row 0 of that head's kv head, of magnitude 0.08-0.14 = 1/(1+8*exp(-s)): the
intended 9-row softmax, with 8 zero rows and the wrong row inserted. Reproduce:
`prov/runb_pv_diagnose.py`, `prov/runb_pv_fixture_diff.py`.

Two consequences for that recipe. `runb_ref_out.npy` (eager torch over all 16 populated rows) is
unreachable by any single-shot run, because cache rows 0..7 are zero in the blob and only the pushed row is
ever written - only the cold expectation is attainable; and build check 1b verifies the power vote, i.e.
the hang fix, so nothing in the recipe tests the numeric contract. Check 1c also shows the skel is not
byte-reproducible (36 bytes differ inside `w8a16_gemv_worker_loop`), so an md5 such as 07c6bf6b cannot be
compared across build directories - verify fixes in the preprocessed source, as 1b does.

**Verified on hardware (2026-09-18 00:49, single shot, exclusive):** their own skel
`hxskel_pv/libhex-htp-skel-v79.so` (07c6bf6b) with `executor_runner_p2` (29e4db17), gate `ctl.pte`
golden 1e4430e9 before the shot, then the contract-correct `zzb_runb_mt1.pte` deployed to `$B/hxmt1`:
`exit d0: ok`, params echoed as `[1,8,1,16,8,128,1035273459,-1,256,0,0,0]`, in1/in2 of 2048 bytes each,
output md5 17aed9fe = the previously verified value, max abs diff 6.1e-05 corr 1.000000 against
`zzb_runb_mt1.pte.ref_causal.npy` and 5.5e-05 corr 1.000000 against `runb_ref_cold.npy`. No watchdog, no
reboot (up 1:59). Their build is numerically correct; only their fixture is stale.

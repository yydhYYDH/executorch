# RPC timeout: why a hang has no error, and how to arm one

A DSP-side hang is currently the worst failure this backend has: the invoke never
returns, so nothing upstream ever sees an error code, no log line explains it, and
only a reboot clears it. This document records the mechanism, the patch that arms
a timeout (already in `hexagon_driver.cpp`, **default-on**), and the three-step
judgment package to run on a device.

## 1. The mechanism

FastRPC's default is **no RPC timeout**. The kernel prints this when it creates the
PD (device log, not from this tree):

```
Created user PD on domain 3, dbg_trace 0x0, enabled attr=> RPC timeout:0, Dbg Mode:N, CRC:N, Unsigned:Y, ...
```

`0` means "wait forever", which matches what we see: the host thread parks in
`fastrpc_wait_for_completion` and never comes back. (The print itself is **not in
the SDK** — see `RPC_TIMEOUT_EXPERIMENT` findings in the task report; `adsprpc.ko`
source is not shipped.)

The app-side knob is one struct and one request id, from the SDK:

- `incs/remote.h:686-702` — `remote_rpc_notif_register_v2 { context, domain, notifier_fn, timeout }`,
  with the field documented as `RPC call timeout (in ms)`. Returns
  `AEE_SUCCESS` / `AEE_EUNSUPPORTED` / `AEE_EBADPARM`.
- `incs/remote.h:949-950` — `FASTRPC_REGISTER_STATUS_NOTIFICATIONS`, passed to
  `remote_session_control(req, &notif, sizeof(notif))`.
- Unit is **ms** (`remote.h:700`, `examples/calculator/src/calculator_main.c:46`).
  `timeout = 0` means "do not enable" — the example branches on `if (timeout_ms)`
  and falls back to the v1 struct (`calculator_test.c:168-172`).
- On expiry the call returns `AEE_EEXPIRED` (`AEEStdErr.h:60` = `AEE_EOFFSET + 0x00C`,
  so **`0x0000000c` on the HLOS side and `0x8000040c` if the DSP form reaches us**),
  the underlying `errno` is `ETIME`, and a `FASTRPC_USERPD_TIMEOUT` notification
  fires (`remote.h:653`; `calculator_test.c:200-202`).
- Ready-made helper to copy from: `utils/examples/pd_status_notification.c:25-46`.

**Platform gate.** `examples/calculator/README.md:371` and
`docs/examples/calculator/index.html:1774` both say: "RPC timeout tests are
currently supported **after Kaanapali** and on CDSP." This device is SM8750, so the
kernel may not implement it. There is no capability bit that answers this —
`is_status_notification_supported()` only queries `STATUS_NOTIFICATION_SUPPORT`
(shared by the v1 and v2 structs), and `enum remote_dsp_attributes` has no timeout
entry. **The registration return code is the test**, which is why the patch logs it
instead of gating on a capability.

Related but different: if a *framework* operation hangs (mmap, unmap, pd kill), the
SDK says SSR is triggered automatically after 5 seconds
(`examples/calculator/README.md:387`, `docs/examples/calculator/index.html`). That
5 s fallback is what makes some stalls self-heal; a plain invoke hang has no such
fallback, which is the gap this patch closes.

## 2. Step 0: what the driver now does

In `backends/hexagon/runtime/hexagon_driver.cpp`, after the unsigned-PD request and
before `htp_ops_open()`, `Open()` calls `ArmRpcTimeout(domain_id_)`:

```c
struct remote_rpc_notif_register_v2 notif;
notif.context = nullptr;
notif.domain = domain_id;              // 3 = cDSP
notif.notifier_fn = RpcStatusNotify;   // prints, does nothing else
notif.timeout = kRpcTimeoutMs;         // 10000
remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, &notif, sizeof(notif));
```

Expected stderr lines and how to read them:

| line | meaning |
|---|---|
| `[hexagon] rpc timeout armed on domain 3: 10000 ms` | kernel accepted the v2 struct — a hang will now expire |
| `[hexagon] rpc timeout NOT armed: registering status notifications with a 10000 ms timeout on domain 3 failed: 0x…` | **this is the platform-gate answer**; see the code table below |
| `[hexagon] pd notification: domain=3 session=N status=USERPD_TIMEOUT(6)` | the callback fired: the timer expired on the DSP side too |
| `[hexagon] execute_command_group_profile exceeded the 10000 ms RPC timeout (AEE_EEXPIRED)` + `failed: 0x0000000c` | judgment A passes |
| `[hexagon] post-timeout probe: original=0x… same_handle=0x… fresh_session=0x…` | the 2-step liveness probe that answers C; read it with §3 Step 1c |

| registration failure code | meaning |
|---|---|
| `0x00000014` / `0x80000414` | `AEE_EUNSUPPORTED` — request/domain not implemented |
| `0x0000006c` / `0x8000046c` | `AEE_EUNSUPPORTEDAPI` — request id unknown to this kernel |
| `0x0000000e` / `0x8000040e` | `AEE_EBADPARM` — struct size/domain rejected |
| `0x00000027` | `AEE_ENOSUCH` — no such domain/process |

**Why 10000 ms and not 0 or 5000.** The timer covers a *single* invoke, not a whole
model, and it must sit above the slowest legitimate one — the >2 s first prefill is
the reason 5 s is too close for comfort; 10 s still turns a permanent hang into a
10-second wait. If a legitimate op ever trips it (visible as a false `0x0c` on a
known-good model), raise `kRpcTimeoutMs`; it is one constant.

**Why it is unconditional.** On a kernel without the feature the call only returns
an error, which we print and ignore — one extra line, no behavior change. A macro
or env switch would mean the diagnostic is absent exactly when it is needed, since
the failure it diagnoses (a hang) gives no chance to set one.

## 3. Step 1: judgment package (device; nothing here runs on the host)

Rebuild and deploy only the **runner** (the skel is untouched, so a timeout-capable
kernel is not required for the artifact to run):

```bash
$CMAKE --build cmake-out-android-arm64-v8a --target executor_runner -j4   # RUNBOOK §2
R=/data/data/com.termux/files/home/csm/qwen3run                          # RUNBOOK §4
scp cmake-out-android-arm64-v8a/executor_runner oneplus13-reverse:$R/
```

**Step 1a — a correct run must stay correct (do this first).** `0x0c` from a healthy
model means the timeout is too small, not that anything else broke:

```bash
cd $R && export LD_LIBRARY_PATH=$PWD/libs:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=$PWD/skel
./executor_runner --model_path <known-good>.pte --num_executions 1 --output_file devout --print_output none 2>&1 | tail -5
```
Pass = `rpc timeout armed` appears **and** `devout-0.bin` still matches the stored
baseline byte-for-byte. Fail = any `0x0000000c` here; raise `kRpcTimeoutMs` and stop.

**Step 1b — A and B on the hanging case.**

```bash
./executor_runner --model_path <hanging>.pte 2>&1 | tee hang.log ; echo "EXIT=$?"
```
- **A**: the run no longer parks forever; ~10 s after the first invoke, stderr shows
  `exceeded the 10000 ms RPC timeout (AEE_EEXPIRED)` and `failed: 0x0000000c`.
  (Today this run prints nothing after `enter d0` and hangs for minutes.)
- **B**: `pd notification: ... status=USERPD_TIMEOUT(6)` appears. If A passes but B
  does not, the timer is host-side only — worth recording, not a failure.
- Also capture `logcat -d | grep -i adsprpc | tail` for the same window.

**Step 1c — C: is the session usable after a timeout?** This decides whether a
timeout can substitute for a reboot. It is answered automatically now:
`ProbeAfterTimeout()` runs on the `AEE_EEXPIRED` path, before the error is returned.

```
[hexagon] post-timeout probe 1/2 (same handle): getInfo -> 0x........
[hexagon] post-timeout probe 2/2 (fresh session on the same PD): open+getInfo -> 0x........
[hexagon] post-timeout probe: original=0x0000000c same_handle=0x........ fresh_session=0x........
```

Both probes always run (probe 2 is unconditional — see below).

| probe 1 (same handle) | probe 2 (fresh session) | conclusion |
|---|---|---|
| `0x0` | `0x0` | the session survived and a fresh one works too: the wedge was transient, no reboot |
| `0x0c` / `0x8000040c` | `0x0` | only the old session was wedged: **close-and-reopen is enough** |
| `0x0c` / `0x8000040c` | `0x0c` / `0x8000040c` | both hang: **the state is outside the PD** (DSP-global) — a reboot is the only known cure |
| `0x8000040d` (`AEE_EBADSTATE`) | any | the handle/PD is already gone: the session *died* rather than wedged |
| any other non-zero | `0x0` | probe 1's error was not a wedge; a fresh session works |

Probe 2 has to run even when probe 1 times out, because probe 1 only answers "is *this*
handle still alive"; the question the probe exists for — "would a fresh session work,
i.e. can a reboot be avoided?" — is exactly the one that matters most when the old
session is dead, and every probe call is bounded by the armed timeout.

**Why the probe is safe at all.** `htp_ops_getInfo` (IDL method 6) takes no VTCM guard
and only reads VTCM/HVX numbers back (`third-party/mnn-htp-ops/src/dsp/commu.cc:274-330`),
so it cannot repeat the work of the call that hung — and, because `Open()` armed the
timeout, even a wedged session makes it return instead of blocking. Without the
timeout this probe would be the very hang it is meant to diagnose, which is why the
two changes ship together.

**Time budget: at most 3 × 10 s = 30 s** for a wasted run — the invoke that expired,
probe 1, probe 2 — since all three can each burn one timeout and nothing is skipped.
A total well under 30 s (the probes usually return in microseconds) is the normal
case; 30 s is the ceiling, not the expected cost.

Probe 2 opens a second handle instead of calling `Close()`: `Close()` frees every live
arena and would corrupt the caller's bookkeeping in the middle of a failure, while a
second handle in the same PD answers the same question and is closed immediately
after. The wedged handle is left for the normal `Close()` path.

If the backend must *retry* in-process after a timeout, that is `hexagon_backend.cpp`
— owned by another agent right now, **not touched here**.

**Why no bad state has to be manufactured first.** Step 0 only registers on the
normal path and reads one return code; Step 1a only checks that a good model stays
good. Both work on a healthy device. Only Step 1b needs a hang, and the existing
repro already produces one. If the hang turns out to require the "bad state" that
only a reboot clears, these same commands still apply — the patch changes nothing
about how the state is entered, only how the wait ends.

## 4. Risks

- A timeout that is too small breaks *correct* runs with `0x0c` — caught by Step 1a,
  fixed by one constant.
- The registration is per-PD, so it covers **every** invoke this host process makes
  on domain 3, not just the stuck one.
- The callback runs on a FastRPC-owned thread and only calls `fprintf`; it must never
  block or issue RPC (a re-entrant call there would wedge the library's own thread).
- Nothing here touches the skel or the .pte, so worst case is one extra log line and
  an error code that no longer hangs.

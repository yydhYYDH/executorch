"""Hard RSS cap for a single export process.

The brief prescribes systemd-run --scope -p MemoryMax= for the real ceiling.
That cannot work on this host: there is no systemd as PID 1 and no session bus,
so every form of systemd-run fails ("System has not been booted with systemd",
"Failed to connect to bus").  This reproduces the property the cap exists for --
an over-budget export kills ITSELF instead of the machine -- and records the
peak so the reading is auditable whether or not the cap fires.
"""
import os
import resource
import sys
import threading
import time

CAP_MB = int(os.environ.get("TTS_RSS_CAP_MB", "1800"))


def peak_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _watchdog():
    while True:
        time.sleep(0.5)
        r = peak_mb()
        if r > CAP_MB:
            print("RSS_CAP_EXCEEDED peak=%d MB cap=%d MB -- aborting self" % (r, CAP_MB), file=sys.stderr)
            sys.stderr.flush()
            os._exit(9)


def start(cap_mb=None):
    global CAP_MB
    if cap_mb:
        CAP_MB = cap_mb
    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()
    print("RSS_CAP_MB %d (watchdog on)" % CAP_MB)
    return t

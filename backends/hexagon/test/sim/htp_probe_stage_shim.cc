// attention_entry.cc and attention_push_kv.cc call this probe, which the device
// build defines in execute_command.cc. That translation unit carries the whole op
// table and its dependencies, so the simulator links the probe alone instead.
//
// It is a trace hook: with no probe buffer installed -- which is the case here --
// the device build returns at the first branch, so a definition that does nothing
// is what the kernels see either way. See test_blob_on_sim.py.
extern "C" void htp_probe_stage(int stage, int a, int b, int c) {
  (void)stage;
  (void)a;
  (void)b;
  (void)c;
}
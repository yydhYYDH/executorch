# Deploying the Paraformer ASR service

This is the end-to-end deployment: the phone runs the acoustic pipeline on the
Hexagon DSP, a small bridge on the server relays it to the browser, and nginx
puts it on HTTPS. All three pieces are described here, with the exact commands
used for the `chengdu` deployment.

## Architecture

```
browser ──https──► nginx (chengdu:443, yiguzhaojin.cn/asr/)
                     │  proxy_pass 127.0.0.1:8020
                     ▼
                   server.py (chengdu, systemd paraformer-web.service, :8020)
                     │  TCP 127.0.0.1:9000
                     ▼
                   ssh -R 9000 (reverse tunnel from the phone)
                     │
                     ▼
                   paraformer_runner --tcp_port 9000 (phone: VAD + ASR + speaker)
```

Three facts drive the whole design:

- The runner binds only `127.0.0.1` on the phone, so it needs a **reverse SSH
  tunnel** to the server. nginx and the bridge never talk to the phone directly.
- The bridge and the runner must share a loopback, which is why the tunnel lands
  on the server's `127.0.0.1:9000` and the bridge connects to it there.
- The runner handles **one connection at a time**; `server.py` serializes
  sessions because the streaming state and HTP resources are shared.

## Phone side

### Build and assets

Build the `paraformer_runner` CMake target with
`-DEXECUTORCH_BUILD_PARAFORMER_RUNNER=ON` (see `../README.md`). On the device the
runner directory needs:

| Asset | What it is |
|---|---|
| `paraformer_runner_spk` | runner with VAD + speaker support |
| `encp_one.pte` | streaming encoder; use the clamped build (`encp_one_min.pte` renamed) |
| `pred15.pte`, `decoder16_first.pte`, `decoder16_next.pte` | predictor and decoders |
| `cmvn.bin`, `tokens.json` | Paraformer frontend / vocabulary |
| `fsmn/fsmn_vad_enc_T20.pte`, `fsmn/cmvn.bin` | FSMN-VAD encoder + CMVN |
| `camppluscpu/campplus_cpu.pte` | CAM++ speaker PTE (CPU / XNNPACK) |
| `skel/` | matching Hexagon skel |

The encoder matters: the earlier `encp_one.pte` overflows fp16 on the q·kᵀ score
matmul of a few far-field chunks and poisons the streaming caches. The fixed
build is the one exported with `torch.minimum(scores, 65504)` (see the encoder
section of `../README.md`); deploy it as `encp_one.pte`.

### Start the runner

`start_tcp_service.sh` starts the VAD + speaker build and picks the clamped
encoder when it is present:

```bash
cd /data/data/com.termux/files/home/csm/execuTorch-ds
PORT=9000 ./start_tcp_service.sh          # prints the pid and the log path
```

It stops only a previous instance on that port, and sets
`LD_LIBRARY_PATH=/system/lib64:/vendor/lib64` and `ADSP_LIBRARY_PATH=$PWD/skel`.

### Reverse tunnel

Keep the runner reachable from the server. On the phone (or a host that can
reach both), run a persistent reverse tunnel:

```bash
autossh -M 0 -NT -R 127.0.0.1:9000:127.0.0.1:9000 chengdu
```

Make it survive reboots with a systemd unit (or the phone's init equivalent):

```ini
[Unit]
Description=Reverse tunnel: phone runner to chengdu
After=network-online.target

[Service]
Environment=AUTOSSH_GATETIME=0
ExecStart=/usr/bin/autossh -M 0 -NT -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:9000:127.0.0.1:9000 chengdu
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Check it from the server: `ss -ltn | grep 127.0.0.1:9000`.

## Server side (chengdu)

### Bridge files

Copy `web/` into `/opt/paraformer-web/`:

```bash
scp web/server.py web/index.html web/app.js web/style.css \
    chengdu:/opt/paraformer-web/
```

`server.py` takes `--runner-host` / `--runner-port` and
`--listen-host` / `--listen-port`; the defaults are the loopback pair used here.

### systemd unit

`/etc/systemd/system/paraformer-web.service`:

```ini
[Unit]
Description=Paraformer web bridge for nginx /asr
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/paraformer-web/server.py \
  --runner-host 127.0.0.1 --runner-port 9000 \
  --listen-host 127.0.0.1 --listen-port 8020
Restart=always
RestartSec=2
WorkingDirectory=/opt/paraformer-web

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now paraformer-web
systemctl status paraformer-web
```

### nginx

`/etc/nginx/snippets/asr.conf`, included from the TLS server block for
`yiguzhaojin.cn`:

```nginx
location = /asr { return 301 /asr/; }

location /asr/ {
    proxy_pass http://127.0.0.1:8020/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection "";
    proxy_buffering off;          # SSE partials and segments must not be buffered
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    client_max_body_size 200m;
}
```

Add `include /etc/nginx/snippets/asr.conf;` inside the `server {}` block of the
site, then `nginx -t && systemctl reload nginx`.

**`proxy_buffering off` is required.** The browser session reads an SSE stream
(`partial`, `segment`, `diarization`, `done`); with buffering on, nginx holds the
events and nothing appears until the request ends.

### Public exposure

This endpoint is **unauthenticated** — anyone who can reach
`https://<host>/asr/` can use it. To take it down, delete
`/etc/nginx/snippets/asr.conf` and the `include` line, then reload nginx.

## Verify

```bash
# bridge is up and sees the runner
curl -s https://<host>/asr/api/health
# {"ok": true, "runner": true}

# one-shot transcription (raw 16 kHz mono signed 16-bit PCM, or a WAV)
curl -s --data-binary @audio.wav -H 'Content-Type: application/octet-stream' \
  https://<host>/asr/api/transcribe
```

Open `https://<host>/asr/` and record from the microphone: partial text appears
live, then each utterance is added as a row with its time span and a colour per
speaker; the final `diarization` event redraws the rows with the offline labels.

## Updating

1. Rebuild/re-export the PTEs on the host and push them and the runner to the
   phone; restart the runner (`PORT=9000 ./start_tcp_service.sh`).
2. Re-copy `web/` to `/opt/paraformer-web/` and `systemctl restart paraformer-web`.

The phone and the server upgrade independently: the bridge only speaks the
runner's TCP protocol, so a new PTE or a new runner binary does not touch the
server, and a new `server.py` does not touch the phone. When the protocol
changes (e.g. a new frame type), update both.

## Limits

- **One active connection**, requests serialized. A second browser session waits.
- The runner caps a session at 256 MiB of audio (~2 hours of 16 kHz mono).
- The tunnel carries all audio to the server and all text back; the DSP work
  stays on the phone. Latency readouts (`rtf`) come from the runner's own timing.

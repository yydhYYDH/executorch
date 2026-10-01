# Paraformer Audio Runner

`paraformer_runner --audio` runs a 16 kHz, mono, signed 16-bit PCM WAV or raw
`.pcm` file on the device. It computes fbank, LFR and CMVN on the device, runs
the encoder, predictor and decoder PTEs chunk by chunk, and prints `TEXT ...`.
It reads the file incrementally and retains fixed-size model state.

Prepare the two non-PTE assets from the FunASR model directory:

```bash
python examples/paraformer/prepare_assets.py MODEL_DIR ASSET_DIR
```

Build the `paraformer_runner` CMake target with
`EXECUTORCH_BUILD_PARAFORMER_RUNNER=ON`, then place the runner, `cmvn.bin`,
`tokens.json`, the four compatible PTEs and the audio file in the same device
directory. On the Android device, run:

```bash
LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH="$PWD/skel" \
  ./paraformer_runner \
  --audio input.wav --cmvn cmvn.bin --tokens tokens.json \
  --model_path encp_one.pte --predictor pred15.pte \
  --decoder_first decoder16_first.pte --decoder_next decoder16_next.pte
```

For repeated requests, replace `--audio input.wav` with `--serve_audio` and
keep the process running. It prints `READY`, then accepts one audio path per
input line. Each request prints `TEXT`, timing, and `DONE 0` on success; send
`QUIT` to exit. All four PTE modules remain loaded between requests.

For a local TCP service, use `--tcp_port` instead. The server binds only to
`127.0.0.1` and keeps the PTE modules loaded:

```bash
./paraformer_runner --tcp_port 9000 --tcp_tmp_dir . \
  --cmvn cmvn.bin --tokens tokens.json \
  --model_path encp_one.pte --predictor pred15.pte \
  --decoder_first decoder16_first.pte --decoder_next decoder16_next.pte
```

On the device, `start_tcp_service.sh` starts the same service with the
expected assets and Hexagon library paths. It stops only an existing
`paraformer_runner_tcp` on the selected port; other HTP users are left alone:

```bash
./start_tcp_service.sh
```

Set `PORT` or `LOG` in the environment to override the defaults (`9000` and
`paraformer_tcp_stream.log`).

The TCP service is streaming and permits only one active connection. Every
message is a one-byte type followed by a little-endian `uint32` payload length
and payload (maximum 256 MiB per session):

| Type | Direction | Payload |
| --- | --- | --- |
| `1` | client -> runner | `START`, empty payload |
| `2` | client -> runner | raw signed 16-bit little-endian mono PCM at 16 kHz |
| `3` | client -> runner | `END`, empty payload |
| `0x80` | runner -> client | start status, little-endian `uint32` |
| `0x81` | runner -> client | UTF-8 partial text |
| `0x83` | runner -> client | segment: `start_ms`, `end_ms`, `speaker`, `text_length`, text |
| `0x82` | runner -> client | final payload: status, elapsed_ms, audio_ms, text_length, text |
| `0x7f` | runner -> client | error status, little-endian `uint32` |

When the FSMN-VAD PTE is passed (`--vad_model` plus `--cmvn_vad`), the runner
also endpoints the stream and emits one `0x83` frame per detected utterance,
carrying its start and end in milliseconds and the text assigned to it. The
Paraformer state is not reset at a boundary; each 600 ms chunk's tokens go to the
segment nearest its center, so the concatenated segment text matches the
continuous transcript. Without the VAD arguments the runner sends no `0x83`
frames and the client sees only partial and final text.

The runner emits a partial-text frame whenever an audio frame produces a new
acoustic chunk. The final response contains recognition wall time and audio
duration in milliseconds; the client computes `rtf = elapsed_ms / audio_ms`.
Requests are serialized because the streaming model state and HTP resources are
shared. A new connection starts a fresh recognition session.

## Browser UI

`web/` contains a dependency-free bridge and a small browser frontend:

- `server.py` serves the UI and relays audio to the runner. It exposes a
  synchronous `POST /api/transcribe` (raw 16 kHz mono signed 16-bit PCM, or a
  WAV whose `data` chunk is used) and a streaming session API:
  `POST /api/session`, `POST /api/session/<id>/audio`, `POST /api/session/<id>/end`,
  `GET /api/session/<id>/events` (Server-Sent Events: `partial` then `done` with
  `elapsed_ms`, `audio_ms`, `rtf`), and `DELETE /api/session/<id>`.
- `index.html`, `app.js`, `style.css` capture microphone PCM live, stream it to
  the runner, and show partial text as it is recognized; uploaded files are
  decoded to 16 kHz mono and sent in one request. Both paths display the
  recognition speed.

```bash
python3 examples/paraformer/web/server.py --runner-port 9000 --listen-port 8020
```

Open `http://127.0.0.1:8020/`. The frontend uses relative URLs, so it also works
behind a reverse proxy at a sub-path such as `/asr/`.

For a phone runner reached over SSH and published through nginx:

```bash
# on the device
./paraformer_runner_tcp --tcp_port 9000 --tcp_tmp_dir .
# on the device: keep the runner reachable on the server's loopback
autossh -M 0 -NT -R 127.0.0.1:9000:127.0.0.1:9000 chengdu
# on the server
python3 server.py --listen-port 8020
```

with an nginx proxy `location /asr/ { proxy_pass http://127.0.0.1:8020/; }`.
Set `proxy_buffering off` so SSE partials stream through. The bridge sends PCM to
the runner in `PCM_CHUNK_BYTES` frames because the streaming runner expects
incremental audio.

The single-delegate encoder `encp_one.pte` takes one fp16 tensor with 2,574,068
values. The runner appends 10 x 560 positional-encoding values and 65 mask
values to the feature/cache input on the device; the encoder then executes as
one DSP call. Position encoding uses float32 before conversion to fp16.

The decoder PTEs have 16 token slots per 600 ms chunk. The runner reports an
error if a predictor chunk exceeds that capacity. The packed-position encoder
PTE stores four base-1024 digits in the last four fp16 input values; the older
single-position PTE is accepted for audio up to position 2048 and rejected
beyond it. Android execution requires the matching Hexagon skel and runtime
libraries from the existing Hexagon build.

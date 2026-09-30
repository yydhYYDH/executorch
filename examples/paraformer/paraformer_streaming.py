#!/usr/bin/env python3
"""End-to-end Paraformer streaming driver for the verified Hexagon PTE stages.

The acoustic frontend runs on the host.  Encoder, predictor, and decoder PTEs
run on a remote OnePlus device through paraformer_runner.  Decoder chunks are
fixed at 16 slots; ``nvalid`` and compacted caches preserve variable token
counts across chunks.  Raw .pcm input is 16 kHz mono signed 16-bit little-endian.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from funasr import AutoModel


CHUNK = [0, 10, 5]
ENC_INPUT = 3 + 10 * 560 + 2 * 50 * 4 * 50 * 128 + 5 * 560
DEC_INPUT = 15 * 512 + 16 * 512 + 16 * 512 * 26 + 32 * 5120 + 1


def command(args):
    for attempt in range(4):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout
        if args[0] not in ("ssh", "rsync") or proc.returncode not in (12, 255) or attempt == 3:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
        print(f"transport retry {attempt + 1}/3: {args[0]} ({proc.returncode})", file=sys.stderr, flush=True)
        time.sleep(3 * (attempt + 1))


class DeviceStages:
    def __init__(self, args, work):
        self.remote = args.remote
        self.device_dir = args.device_dir
        self.runner = args.runner
        self.local = Path(work)
        self.remote_prefix = f"{self.remote}:{self.device_dir}/"
        self.models = {
            "enc": args.encoder,
            "pred": args.predictor,
            "dec_first": args.decoder_first,
            "dec_next": args.decoder_next,
        }
        for name, path in self.models.items():
            command(["rsync", "-a", "--checksum", "--partial", "--inplace", path, self.remote_prefix])

    def run(self, stage, data, output_count, tag):
        input_name = f"{tag}.in.bin"
        output_prefix = f"{tag}.out"
        local_input = self.local / input_name
        data.astype(np.float16, copy=False).tofile(local_input)
        command(["rsync", "-a", "--checksum", "--partial", "--inplace", str(local_input), self.remote_prefix])
        model = self.models[stage]
        remote_cmd = (
            f"cd {self.device_dir} && timeout 120s env "
            f"LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 "
            f"ADSP_LIBRARY_PATH=$PWD/skel ./{self.runner} "
            f"--model_path {Path(model).name} --inputs {input_name} "
            f"--output_file {output_prefix}"
        )
        command(["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
                 "-o", "ServerAliveCountMax=3", self.remote, remote_cmd])
        command(["rsync", "-a", "--checksum", "--partial", "--inplace",
                 f"{self.remote}:{self.device_dir}/{output_prefix}-*.bin", str(self.local)])
        outputs = []
        for i in range(output_count):
            name = f"{output_prefix}-{i}.bin"
            raw = (self.local / name).read_bytes()
            # Stage output dtypes are known from the verified PTE contracts.
            if stage == "pred" and i in (0, 2, 3):
                dtype = np.float32
            else:
                dtype = np.float16
            outputs.append(np.frombuffer(raw, dtype=dtype).copy())
        bad = [i for i, value in enumerate(outputs) if not np.isfinite(value).all()]
        if bad:
            raise RuntimeError(f"{stage} {tag} produced non-finite outputs: {bad}")
        return outputs


def capture_chunks(model, wav_path):
    encoder = model.model.encoder.eval()
    captured = []
    original = encoder.forward_chunk

    def capture(*args, **kwargs):
        cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
        opt = cache.get("opt") if cache else None
        captured.append({
            "xs": args[0].detach().clone(),
            "feats": cache["feats"].detach().clone(),
            "start_idx": int(cache["start_idx"]),
            "ncache": 0 if opt is None else int(opt[0]["k"].shape[2]),
        })
        result = original(*args, **kwargs)
        return result

    encoder.forward_chunk = capture
    try:
        audio = (np.fromfile(wav_path, dtype=np.int16).astype(np.float32) / 32768.0
                 if wav_path.suffix.lower() == ".pcm" else str(wav_path))
        model.generate(
            input=audio, chunk_size=CHUNK,
            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
            is_final=True,
        )
    finally:
        encoder.forward_chunk = original
    return captured


def encode_input(chunk, enc_state):
    buf = np.zeros(ENC_INPUT, dtype=np.float16)
    n = chunk["xs"].shape[1]
    buf[0:3] = (chunk["start_idx"], n, chunk["ncache"])
    begin = 3
    buf[begin:begin + n * 560] = chunk["xs"].to(torch.float16).numpy().reshape(-1)
    begin += 10 * 560
    buf[begin:begin + enc_state.size] = enc_state
    begin += 2 * 50 * 4 * 50 * 128
    buf[begin:] = chunk["feats"].to(torch.float16).numpy().reshape(-1)
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="WAV or 16 kHz mono signed int16 .pcm")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/yydh/models/paraformer-zh-streaming"))
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--decoder-first", required=True)
    parser.add_argument("--decoder-next", required=True)
    parser.add_argument("--remote", default="oneplus13-reverse")
    parser.add_argument("--device-dir", default="/data/data/com.termux/files/home/csm/execuTorch-ds")
    parser.add_argument("--runner", default="paraformer_runner")
    parser.add_argument("--checkpoint", type=Path, help="save state after each completed chunk and resume")
    parser.add_argument("--dry-run", action="store_true", help="capture frontend chunks without using the phone")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.audio.suffix.lower() == ".pcm" and args.audio.stat().st_size % 2:
        parser.error("raw PCM must have an even number of bytes")
    with tempfile.TemporaryDirectory(prefix="paraformer_hexagon_") as tmp:
        model = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
        chunks = capture_chunks(model, args.audio)
        print(f"captured {len(chunks)} encoder chunks: "
              f"{[(c['start_idx'], c['xs'].shape[1], c['ncache']) for c in chunks]}")
        if args.dry_run:
            return
        device = DeviceStages(args, tmp)
        enc_state = np.zeros(2 * 50 * 4 * 50 * 128, dtype=np.float16)
        predictor_alpha = np.zeros(1, dtype=np.float16)
        predictor_hidden = np.zeros(512, dtype=np.float16)
        decoder_fsmn = np.zeros(16 * 512 * 26, dtype=np.float16)
        decoder_opt = np.zeros(32 * 5120, dtype=np.float16)
        tokens = []
        decoder_started = False
        start_index = 0
        audio_hash = hashlib.sha256(args.audio.read_bytes()).hexdigest()
        model_key = "\n".join(
            f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in (Path(args.encoder), Path(args.predictor),
                         Path(args.decoder_first), Path(args.decoder_next))
        )
        if args.checkpoint and args.checkpoint.exists():
            with np.load(args.checkpoint, allow_pickle=False) as saved:
                if (str(saved["audio_hash"]) != audio_hash or
                        str(saved["model_key"]) != model_key or
                        int(saved["chunk_count"]) != len(chunks)):
                    raise RuntimeError("checkpoint does not match audio, PTEs, or frontend chunk count")
                start_index = int(saved["last_index"]) + 1
                enc_state = saved["enc_state"].copy()
                predictor_alpha = saved["predictor_alpha"].copy()
                predictor_hidden = saved["predictor_hidden"].copy()
                decoder_fsmn = saved["decoder_fsmn"].copy()
                decoder_opt = saved["decoder_opt"].copy()
                decoder_started = bool(saved["decoder_started"])
                tokens = saved["tokens"].tolist()
            print(f"resuming at chunk {start_index}", flush=True)
        for index in range(start_index, len(chunks)):
            chunk = chunks[index]
            enc_input = encode_input(chunk, enc_state)
            enc = device.run("enc", enc_input, 3, f"enc_{index}")
            h = enc[0]
            enc_state = enc[1]
            pred_input = np.zeros(8196, dtype=np.float16)
            pred_input[:7680] = h
            pred_input[7680] = predictor_alpha[0]
            pred_input[7681:8193] = predictor_hidden
            pred_input[8194] = 1.0 if index == len(chunks)-1 else 0.0
            pred = device.run("pred", pred_input, 4, f"pred_{index}")
            valid = int(np.count_nonzero(pred[1]))
            if not 0 <= valid <= 16:
                raise RuntimeError(f"predictor returned {valid} tokens; fixed decoder capacity is 16")
            predictor_alpha, predictor_hidden = pred[2].astype(np.float16), pred[3].astype(np.float16)
            if valid:
                dec_input = np.zeros(DEC_INPUT, dtype=np.float16)
                dec_input[:7680] = h
                dec_input[7680:7680 + valid * 512] = pred[0][:valid * 512].astype(np.float16)
                dec_input[7680 + 16 * 512:7680 + 16 * 512 + decoder_fsmn.size] = decoder_fsmn
                dec_input[7680 + 16 * 512 + decoder_fsmn.size:-1] = decoder_opt
                dec_input[-1] = valid
                stage = "dec_next" if decoder_started else "dec_first"
                dec = device.run(stage, dec_input, 49, f"dec_{index}")
                decoder_started = True
                logits = dec[0].reshape(16, 8404)[:valid]
                tokens.extend(logits.argmax(-1).tolist())
                decoder_fsmn = np.concatenate([x for x in dec[1:17]])
                decoder_opt = np.concatenate([x for x in dec[17:]])
            print(f"chunk {index}: valid={valid}, tokens={tokens[-valid:] if valid else []}", flush=True)
            if args.checkpoint:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pending = args.checkpoint.with_name(args.checkpoint.name + ".pending.npz")
                np.savez(pending, audio_hash=audio_hash, model_key=model_key,
                         chunk_count=len(chunks),
                         last_index=index, enc_state=enc_state,
                         predictor_alpha=predictor_alpha, predictor_hidden=predictor_hidden,
                         decoder_fsmn=decoder_fsmn, decoder_opt=decoder_opt,
                         decoder_started=decoder_started, tokens=np.asarray(tokens, dtype=np.int32))
                os.replace(pending, args.checkpoint)
        eos = tokens.index(2) if 2 in tokens else len(tokens)
        print(model.kwargs["tokenizer"].decode(tokens[:eos]))


if __name__ == "__main__":
    main()

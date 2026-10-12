BL_OVERRIDE = globals().get("BL_OVERRIDE")
import json, os, torch

AL = [a / 32.0 for a in json.load(open(S + 'alpha_fc.json'))]
M = [v / 32.0 for v in json.load(open(S + 'fc_inmag.json'))]
BET = [max(1.0, v / 65504.0) if v > 0 else 1.0
       for v in json.load(open(S + 'ffn_mag_fc.json'))]
BET[38] *= 1.01
W2_BET = {j: float(os.environ.get(f"EXTRA_W2_BET{j}", os.environ.get("EXTRA_W2_ALL", "1.0")))
          for j in range(len(M))}
NL = len(M)
CB = 50
DBG = globals().get('DBG') or {}
POSITION_DIGITS = globals().get('POSITION_DIGITS', False)
FUSE_MASK = globals().get('FUSE_MASK', False)
PREPARED_ENCODER = globals().get('PREPARED_ENCODER', False)
bl = BL_OVERRIDE if BL_OVERRIDE is not None else list(E.encoders0) + list(E.encoders)
NL = len(bl) if BL_OVERRIDE is not None else NL
print('M[0]=%.2f M[1]=%.2f M[%d]=%.2f  folded=%d' % (
    M[0], M[1], NL - 1, M[NL - 1], sum(1 for v in BET if v > 1.0)), flush=True)

TRACE = []


def _att_fc(self, x, cache=None, chunk_size=None, look_back=0, mask=None, vm=None):
    q_h, k_h, v_h, v = self.forward_qkv(x)
    if cache is not None:
        k_s = k_h[:, :, : -(chunk_size[2]), :]
        v_s = v_h[:, :, : -(chunk_size[2]), :]
        k_h = torch.cat((cache['k'], k_h), dim=2)
        v_h = torch.cat((cache['v'], v_h), dim=2)
        cache['k'] = torch.cat((cache['k'][:, :, chunk_size[1]:, :], k_s), dim=2)
        cache['v'] = torch.cat((cache['v'][:, :, chunk_size[1]:, :], v_s), dim=2)
    fsmn = self.forward_fsmn(v * vm, None)
    s = torch.matmul((q_h * self.d_k ** -0.5).float(),
                     k_h.transpose(-2, -1).float())
    # Keep the scores inside fp16: the store is the overflow point. Use the
    # binary minimum, not clamp: the DSP's unary CLAMP deliberately passes an
    # inf/NaN through (unary_ops.cc:534), while `a < b ? a : b` turns +inf into
    # 65504. It is a no-op for every score that already fits, and exact where a
    # row's only overflowed score is its maximum -- softmax subtracts that
    # maximum, so its absolute value never reaches the result.
    s = torch.minimum(s, torch.tensor(65504.0, dtype=s.dtype, device=s.device))
    if mask is not None:
        m = mask.unsqueeze(1).lt(0.5) if FUSE_MASK else mask.unsqueeze(1).eq(0)
        if FUSE_MASK:
            s = torch.where(m, self.mask_floor, s)
            a = torch.where(m, self.mask_zero, torch.softmax(s, dim=-1)).to(DT)
        else:
            s = s.masked_fill(m, -float('inf'))
            a = torch.softmax(s, dim=-1).masked_fill(m, 0.0).to(DT)
    else:
        a = torch.softmax(s, dim=-1).to(DT)
    x = torch.matmul(a, v_h).transpose(1, 2).contiguous().view(q_h.shape[0], -1, self.h * self.d_k)
    return self.linear_out(x.to(DT)) + fsmn, cache


def _mkfwd(j, B):
    al, bt = M[j], BET[j] * W2_BET.get(j, 1.0)
    nxt = M[j + 1] if j + 1 < NL else M[j]
    conv = al / nxt
    rec = [0.0]

    def f(x, cache=None, chunk_size=None, look_back=0, mask=None, vm=None, ncache=0):
        n = (x.to(DT) / 32.0) if j == 0 else x.to(DT)
        att, cache = B.self_attn.forward_chunk(
            B.norm1(n.to(B.norm1.weight.dtype)), cache, chunk_size, look_back, mask, vm)
        scale = torch.tensor(al, dtype=torch.float32)
        a = ((att.float() / scale) / 32.0).to(DT)
        y = (n + a).to(DT) if B.in_size == B.size else a.to(DT)
        yn = B.norm2(y.to(B.norm2.weight.dtype))
        ff = B.feed_forward(yn).to(DT)
        if DBG.get('all'):
            DBG.setdefault('ns', []).append(n)
            DBG.setdefault('g%d' % n.shape[-1], []).append(n)
        if DBG.get('j') == j:
            DBG['n'] = n
            DBG['a'] = a
            DBG['y'] = y
            DBG['yn'] = yn.to(DT)
            DBG['ff'] = ff
        of = (((ff.float() / scale) / 32.0).to(DT) * bt).to(DT)
        o = ((y + of).to(DT) * conv).to(DT)
        if DBG.get('fix'):
            DBG.setdefault('o512', []).append(o.to(DT))
        if TRACE:
            rec[0] = float(o.float().abs().max())
        return o, cache
    f.trace = rec
    return f


class Lin1x1(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.conv = torch.nn.Conv1d(lin.in_features, lin.out_features, 1,
                                    bias=lin.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(lin.weight.unsqueeze(-1))
            if lin.bias is not None:
                self.conv.bias.copy_(lin.bias)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


for j, B in enumerate(bl):
    B.self_attn.forward_chunk = _att_fc.__get__(B.self_attn, type(B.self_attn))
    B.forward_chunk = _mkfwd(j, B)
    with torch.no_grad():
        w1 = B.feed_forward.w_1
        if isinstance(w1, torch.nn.Linear) and BET[j] > 1.0:
            w1.weight.div_(BET[j])
            w1.bias.div_(BET[j])
        if W2_BET.get(j, 1.0) > 1.0:
            B.feed_forward.w_2.weight.div_(W2_BET[j])
            B.feed_forward.w_2.bias.div_(W2_BET[j])

E.embed.to(DT)
E.after_norm.to(DT)
for B in bl:
    B.self_attn.to(DT)
    B.feed_forward.to(DT)
    B.norm1.to(DT)
    B.norm2.to(DT)
    if FUSE_MASK:
        B.self_attn.register_buffer('mask_floor', torch.tensor(-float('inf'), dtype=torch.float32))
        B.self_attn.register_buffer('mask_zero', torch.tensor(0.0, dtype=torch.float32))


class ChunkEnc(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        if FUSE_MASK:
            self.register_buffer('mask_zero', torch.tensor(0.0, dtype=DT))

    def forward(self, buf, start_idx_input=None):
        o_ct = 0
        o_xs = 3
        o_opt = o_xs + T * 560
        o_ft = o_opt + 2 * NL * 4 * CB * 128
        xs = buf[0, o_xs:o_opt].view(1, T, 560)
        opt = buf[0, o_opt:o_ft].view(2 * NL, 4, CB, 128)
        feats = buf[0, o_ft:o_ft + (CH[0] + CH[2]) * 560].view(1, CH[0] + CH[2], 560)
        ctrl = buf[0, o_ct:o_ct + 3]
        if PREPARED_ENCODER:
            start_idx = None
        elif POSITION_DIGITS:
            digits = buf[0, o_ft + (CH[0] + CH[2]) * 560:
                         o_ft + (CH[0] + CH[2]) * 560 + 4].to(torch.int64)
            start_idx = (digits[0] + digits[1] * 1024 +
                         digits[2] * 1048576 + digits[3] * 1073741824)
        else:
            start_idx = (ctrl[0].to(torch.int64) if start_idx_input is None
                         else start_idx_input.to(torch.int64).reshape(()))
        nvalid = ctrl[1].to(torch.int64) if not PREPARED_ENCODER else None
        ncache = ctrl[2].to(torch.int64) if not PREPARED_ENCODER else 0
        x = (xs.to(DT) * (self.enc.output_size() ** 0.5)).to(DT)
        nf = CH[0] + CH[2]
        if PREPARED_ENCODER:
            o_pos = o_ft + nf * 560
            position_encoding = buf[0, o_pos:o_pos + T * 560].view(1, T, 560)
            mask = buf[0, o_pos + T * 560:o_pos + T * 560 + CB + nf + T].view(1, 1, CB + nf + T)
            cur = mask[:, 0, CB:]
        else:
            pos = start_idx.reshape(1, 1) + torch.arange(1, T + 1).view(1, T)
            position_encoding = self.enc.embed.encode(pos, x.shape[2], torch.float32).to(DT)
            cur = torch.cat((torch.ones(1, nf, dtype=DT),
                         (torch.arange(1, T + 1, dtype=DT) <= nvalid.to(DT)).to(DT).view(1, T)),
                        dim=1)
            past = (torch.arange(1, CB + 1, dtype=DT) > (CB - ncache).to(DT)).to(DT).view(1, -1)
            mask = torch.cat((past, cur), dim=1).view(1, 1, CB + nf + T)
        x = (x + position_encoding).to(DT)
        x = (x * cur[:, nf:].unsqueeze(-1).to(DT)).to(DT)
        ov = torch.cat((feats, x), dim=1)
        vmask = cur.view(1, -1, 1).to(DT)
        parts = opt.unbind(0)
        new = [{'k': parts[2 * j].unsqueeze(0), 'v': parts[2 * j + 1].unsqueeze(0)}
               for j in range(len(parts) // 2)]
        h = ov
        h_copies = []
        kv_copies = []
        for j, B in enumerate(bl):
            h, new[j] = B.forward_chunk(h, new[j], CH, LB, mask, vmask, ncache)
            h_copies.append(h.to(DT))
            kv_copies.extend((new[j]['k'].to(DT), new[j]['v'].to(DT)))
            h = torch.where(vmask > 0, h, self.mask_zero if FUSE_MASK else torch.zeros_like(h)).to(DT)
        h_pre = h
        g49_in = DBG['g512'][48].clone() if (DBG.get('all') and len(DBG.get('g512', [])) > 48) else h_pre
        if self.enc.normalize_before:
            h = self.enc.after_norm(h)
        st = torch.stack([t for d in new for t in (d['k'], d['v'])], dim=0)
        if DBG.get('j') is not None and DBG.get('yn') is not None:
            return (h, st.squeeze(1), ov[:, T:, :], DBG['n'], DBG['a'],
                    DBG['y'], DBG['yn'], DBG['ff'])
        if DBG.get('fix'):
            _o49 = torch.stack([h_copies[-1], h_copies[-1]], dim=0)[0].to(DT)
            _o49 = torch.where(vmask > 0, _o49, self.mask_zero if FUSE_MASK else torch.zeros_like(_o49)).to(DT)
            _stk = torch.stack(kv_copies, dim=0).squeeze(1)
            return (self.enc.after_norm(_o49).reshape(-1).to(DT), _stk.reshape(-1).to(DT), ov[:, T:, :].reshape(-1).to(DT))
        if DBG.get('pre'):
            return (h.reshape(-1).to(DT), h_pre.reshape(-1).to(DT))
        return h, st.squeeze(1), ov[:, T:, :]


W = ChunkEnc(E).eval()

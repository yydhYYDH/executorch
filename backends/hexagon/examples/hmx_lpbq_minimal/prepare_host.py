import ctypes
import struct

import numpy as np
from verify import reference


def prepare_cases(library, cases):
    lib = ctypes.CDLL(str(library))
    prepare = lib.hmx_lpbq_prepare
    prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint16,
        ctypes.c_float,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    prepare.restype = ctypes.c_int
    data = bytearray(struct.pack("<I", len(cases)))
    for a, w, scales, sa, sq, za, zq, global_scale in cases:
        w = np.ascontiguousarray(w, dtype="i1")
        scales = np.ascontiguousarray(scales, dtype="<f4")
        packed = np.empty(8192, dtype="i1")
        table = np.empty(256, dtype="<u4")
        result = prepare(
            w.ctypes.data,
            scales.ctypes.data,
            float(global_scale),
            float(sa),
            za,
            float(sq),
            zq,
            packed.ctypes.data,
            table.ctypes.data,
        )
        if result:
            raise ValueError("Unsupported LPBQ parameters")
        expected_table = reference((a, w, scales, sa, sq, za, zq, global_scale))[2]
        np.testing.assert_array_equal(table, expected_table)
        multiples = np.clip(np.rint(scales / (scales.max(axis=1)[:, None] / 16)), 1, 16)
        expected_weights = w.astype(np.int32) * np.repeat(multiples, 32, axis=1).T
        for block in range(4):
            # [N32 group,K4 group,N lane,K lane] is the native byte-weight order.
            expected_block = expected_weights[block * 32 : (block + 1) * 32]
            expected_block = expected_block.reshape(8, 4, 2, 32).transpose(2, 0, 3, 1)
            np.testing.assert_array_equal(
                packed[block * 2048 : (block + 1) * 2048], expected_block.reshape(-1)
            )
        data.extend(packed.tobytes() + table.tobytes())
    return data

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.examples.models.llama.model import Llama2Model
from executorch.examples.models.qwen3.convert_weights import convert_weights


class Qwen3Model(Llama2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen3VLEmbedsModel(Qwen3Model):
    """The Qwen3-VL text tower with the embedding lookup and RoPE moved out.

    Qwen3-VL splices image features into the text tower's input embeddings, which
    a token-id program cannot express, and rotates them with M-RoPE, where each
    frequency pair takes its position from the (t, h, w) of the token in the
    visual grid. The caller therefore supplies both the embedding and the
    frequencies, one position per call, and the KV cache carries the sequence.
    input_pos stays an input because it indexes the cache and the attention mask.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            not self.model_.params.apply_embedding
        ), "set apply_embedding to false in the params json"
        assert self.model_.params.rope_from_input, "set rope_from_input to true"

    def get_example_inputs(self):
        head_dim = self.model_.params.head_dim
        seq_len = self.model_.params.example_seq_len
        dim = self.model_.params.dim
        inputs = {
            "input_pos": torch.arange(seq_len, dtype=torch.long),
            # 2D, the shape a freqs_cos[input_pos] gather produces
            "rope_cos": torch.ones(seq_len, head_dim, dtype=torch.float16),
            "rope_sin": torch.zeros(seq_len, head_dim, dtype=torch.float16),
        }
        for i in range(self.model_.params.deepstack_inputs):
            inputs["deepstack_%d" % i] = torch.zeros(
                1, seq_len, dim, dtype=torch.float16
            )
        return torch.ones(1, seq_len, dim, dtype=torch.float16), inputs


__all__ = [
    "Qwen3Model",
    "Qwen3VLEmbedsModel",
    "convert_weights",
]

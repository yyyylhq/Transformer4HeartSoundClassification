import enum
import chex
import numpy as np
import jax.numpy as jnp
from typing import Union

class PositionalEncodings(enum.Enum):
    NONE = 0
    SIN = 1
    RELATIVE = 2
    ROTARY = 3
    LEARNT = 4
    ALIBI = 5

@chex.dataclass
class SinPEParams:
    max_time: int = 10_000

RotaryPEParams = SinPEParams
RelativePEParams = SinPEParams

@chex.dataclass
class LearntPEParams:
    max_sequence_length: int

PositionalEncodingsParams = Union[
    SinPEParams,
    RelativePEParams,
    RotaryPEParams,
    LearntPEParams
]

POS_ENC_TABLE = {
    'NONE': PositionalEncodings.NONE,
    'SIN': PositionalEncodings.SIN,
    'RELATIVE': PositionalEncodings.RELATIVE,
    'ROTARY': PositionalEncodings.ROTARY,
    'LEARNT': PositionalEncodings.LEARNT,
    'ALIBI': PositionalEncodings.ALIBI
}

POS_ENC_PARAMS_TABLE = {
    'NONE': SinPEParams,
    'SIN_COS': None,
    'ALIBI': SinPEParams,
    'RELATIVE': RelativePEParams,
    'ROTARY': RotaryPEParams,
    'LEARNT': LearntPEParams
}

def sinusoidal_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_time_scale: int = 10_000,
    add_negative_side: bool = False
) -> jnp.ndarray:
    if hidden_size % 2 != 0:
        raise ValueError("The feature dimension should be even for sinusoidal position encodings.")

    freqs = jnp.arange(0, hidden_size, 2, dtype=jnp.float32)
    inv_freq = max_time_scale ** (- freqs / hidden_size)

    pos_seq = jnp.arange(
        start=-sequence_length if add_negative_side else 0,
        stop=sequence_length,
        dtype=jnp.float32
    )

    sinusoid_inp = jnp.einsum("i,j->ij", pos_seq, inv_freq)

    return jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)


def apply_rotary_encoding(
    x: jnp.ndarray,
    position: jnp.ndarray,
    max_time: int = 10_000
):

    def _rope_kernel(n: int, dtype=jnp.float32) -> jnp.ndarray:
        kernel = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            if i % 2 == 0:
                kernel[i, i + 1] = 1
            else:
                kernel[i, i - 1] = -1

        return jnp.array(kernel)

    if x.shape[-1] % 2 != 0:
        raise ValueError("The feature dimension should be even for sinusoidal position encodings.")

    freq_seq = jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    freq_seq = freq_seq / (x.shape[-1] // 2)
    inv_freq = max_time**-freq_seq
    inv_freq = jnp.repeat(inv_freq, 2, 0)

    #position.shape = (1, T), t.shape = (1, T, 1, D)
    t = position[:, :, None, None] * inv_freq[None, None, None, :]

    x_rot = jnp.einsum("bthd,dD->bthD", x, _rope_kernel(x.shape[-1], x.dtype))

    #a = x * jnp.cos(t).astype(x.dtype) + jnp.sin(t).astype(x.dtype) * x_rot
    #print(a.shape)
    return x * jnp.cos(t).astype(x.dtype) + jnp.sin(t).astype(x.dtype) * x_rot

def compute_attention_with_relative_encodings(
    queries: chex.Array,
    keys: chex.Array,
) -> chex.Array:

    batch_size, k_seq_len, num_heads, num_hiddens = keys.shape
    hiddens = num_hiddens * num_heads

    # First compute the content logits.
    content_bias = hk.get_parameter(
        name='relpos_contentbias',
        shape=[num_heads, num_hiddens],
        init=hk.initializers.RandomNormal(stddev=0.02)
    )
    content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

    positional_encodings = sinusoid_position_encoding(
        sequence_length=k_seq_len,
        hidden_size=hiddens,
        add_negative_side=True
    )
    positional_encodings = jnp.broadcast_to(
        positional_encodings,
        (batch_size,) + positional_encodings.shape
    )
    relative_keys = hk.Linear(hiddens, with_bias=False)(positional_encodings)
    relative_keys = jnp.reshape(
        relative_keys,
        positional_encodings.shape[:-1] + (num_heads, num_hiddens)
    )

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02),
  )
  relative_logits = jnp.einsum(
      'bthd,bThd->bhtT', queries + relative_bias, relative_keys
  )
  # We shift the relative logits instead of the positional encoding matrix as
  # described in Appendix B of the paper (https://arxiv.org/pdf/1901.02860.pdf).
  relative_logits = relative_shift(
      relative_logits, attention_length=content_logits.shape[-1], causal=causal
  )
  assert content_logits.shape == relative_logits.shape
  return content_logits + relative_logits

if __name__ == "__main__":
    #sinusoidal_position_encoding(100, 512)


    x = jnp.zeros((256, 100, 1, 512))
    pos = jnp.arange(100)[None, :]


    apply_rotary_encoding(x, pos)

import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

def sinusoidal_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_time_scale: int = 10_000,
    add_negative_side: bool = False
) -> jnp.ndarray:
    """
    return: if add_negative_side == False, shape=(sequence_length, hidden_size)
            if add_negative_side == True, shape=(2 * sequence_length, hidden_size)
    """

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
    """
    x: shape=(b, t, d) or (b, t, h, d)
    position: shape=(1, t)
    """

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

    freq_seq = jnp.arange(x.shape[-1] // 2, dtype=jnp.float32) / (x.shape[-1] // 2)

    inv_freq = max_time ** (-freq_seq)

    inv_freq = jnp.repeat(inv_freq, repeats=2, axis=0)

    #position.shape = (1, T), t.shape = (1, T, D)
    t = None
    x_rot = None
    if x.ndim == 3:
        t = position[:, :, None] * inv_freq[None, None, :]
        x_rot = jnp.einsum("btd,dD->btD", x, _rope_kernel(x.shape[-1], x.dtype))
    elif x.ndim == 4:
        t = position[:, :, None, None] * inv_freq[None, None, None, :]
        x_rot = jnp.einsum("bthd,dD->bthD", x, _rope_kernel(x.shape[-1], x.dtype))
    else:
        raise ValueError("The feature numbers of dimension should be 3(b, t, d) or 4(b, t, h, d).")


    return x * jnp.cos(t).astype(x.dtype) + jnp.sin(t).astype(x.dtype) * x_rot

class TransformerEncoderLayer(nn.Module):
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        _, _, d = x.shape

        x = nn.MultiHeadAttention(num_heads=self.nhead, qkv_features=d, deterministic=not training)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(4 * d)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(d)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.LayerNorm()(x)

        return x

class TransformerEncoderLayerwithRoPE(nn.Module):
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        b, l, d = x.shape
        if d % self.nhead != 0:
            raise ValueError("The feature dimension should be even for the numbers of head.")

        q = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)
        k = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)
        v = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)

        q = apply_rotary_encoding(q, jnp.arange(x.shape[1])[jnp.newaxis, :])
        k = apply_rotary_encoding(k, jnp.arange(x.shape[1])[jnp.newaxis, :])


        if training:
            x = nn.dot_product_attention(q, k, v, dropout_rng=self.make_rng("dropout"), dropout_rate=self.dropout_rate, deterministic=not training).reshape(b, l, d)
        else:
            x = nn.dot_product_attention(q, k, v, deterministic=not training).reshape(b, l, d)

        x = nn.Dense(d)(x)

        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(4 * d)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(d)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.LayerNorm()(x)

        return x

class TransformerEncoderLayerwithRelativePE(nn.Module):
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):

        def _relative_shift(
            logits: jax.Array,
            attention_length: int,
        ) -> jax.Array:

            def _rel_shift_inner(logits: jax.Array, attention_length: int) -> jax.Array:
                if logits.ndim != 2:
                    raise ValueError('`logits` needs to be an array of dimension 2.')
                tq, total_len = logits.shape
                assert total_len == tq + attention_length
                logits = jnp.reshape(logits, [total_len, tq])
                logits = jax.lax.slice(logits, (1, 0), logits.shape)  # logits[1:]
                logits = jnp.reshape(logits, [tq, total_len - 1])
                logits = jax.lax.slice(logits, (0, 0), (tq, attention_length))

                return logits

            fn = lambda t: _rel_shift_inner(t, attention_length)

            return jax.vmap(jax.vmap(fn))(logits)

        b, l, d = x.shape
        if d % self.nhead != 0:
            raise ValueError("The feature dimension should be even for the numbers of head.")

        q = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)
        k = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)
        v = nn.Dense(d)(x).reshape(b, l, self.nhead, -1)

        content_bias = self.param("content_bias", nn.initializers.normal(), (self.nhead, d // self.nhead))
        relative_bias = self.param("relative_bias", nn.initializers.normal(), (self.nhead, d // self.nhead))

        content_logits = jnp.einsum('bthd,bThd->bhtT', q + content_bias, k)

        positional_encodings = sinusoidal_position_encoding(
            sequence_length=l,
            hidden_size=d,
            add_negative_side=True,
        )
        positional_encodings = jnp.broadcast_to(
            positional_encodings, (b,) + positional_encodings.shape
        )

        relative_keys = nn.Dense(d, use_bias=False)(positional_encodings)
        relative_keys = jnp.reshape(
            relative_keys, positional_encodings.shape[:-1] + (self.nhead, d // self.nhead)
        )

        relative_logits = jnp.einsum(
            "bthd,bThd->bhtT", q + relative_bias, relative_keys
        )

        relative_logits = _relative_shift(
            relative_logits, attention_length=content_logits.shape[-1]
        )

        attention_logits = content_logits + relative_logits
        attention = nn.softmax(attention_logits)

        if training:
            keep_prob = 1.0 - self.dropout_rate
            dropout_shape = tuple([1] * (attention.ndim - 2)) + attention.shape[-2:]
            keep = jax.random.bernoulli(self.make_rng("dropout"), keep_prob, dropout_shape)
            multiplier = keep.astype(jnp.float32) / jnp.asarray(keep_prob, dtype=jnp.float32)

            attention = attention * multiplier

        x = jnp.einsum(
            "bhtT,bThd->bthd", attention, v
        ).reshape(b, l, d)

        x = nn.Dense(d)(x)

        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(4 * d)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(d)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.LayerNorm()(x)

        return x

class T4HSC(nn.Module):
    num_classes: int = 2
    hidden_dim: int = 2048
    embed_dim: int = 512

    num_layer: int = 6
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.embed_dim)(x)

        class_token_embedding = self.param("class_token", nn.initializers.normal(), (1, 1, self.embed_dim))

        class_token = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))
        x = jnp.concatenate([class_token, x], axis=1)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(nhead=self.nhead, dropout_rate=self.dropout_rate)(x, training=training)

        x = x[:, 0, :]
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)

        return x

class T4HSCwithSinPE(nn.Module):
    num_classes: int = 2
    hidden_dim: int = 2048
    embed_dim: int = 512

    num_layer: int = 6
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.embed_dim)(x)

        class_token_embedding = self.param("class_token", nn.initializers.normal(), (1, 1, self.embed_dim))

        class_token = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))
        x = jnp.concatenate([class_token, x], axis=1)

        x = x + sinusoidal_position_encoding(x.shape[-2], x.shape[-1])

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(nhead=self.nhead, dropout_rate=self.dropout_rate)(x, training=training)

        x = x[:, 0, :]
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)

        return x

class T4HSCwithRoPE(nn.Module):
    num_classes: int = 2
    hidden_dim: int = 2048
    embed_dim: int = 512

    num_layer: int = 6
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.embed_dim)(x)

        class_token_embedding = self.param("class_token", nn.initializers.normal(), (1, 1, self.embed_dim))

        class_token = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))

        x = jnp.concatenate([class_token, x], axis=1)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayerwithRoPE(nhead=self.nhead, dropout_rate=self.dropout_rate)(x, training=training)

        x = x[:, 0, :]
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)

        return x

class T4HSCwithRelativePE(nn.Module):
    num_classes: int = 2
    hidden_dim: int = 2048
    embed_dim: int = 512

    num_layer: int = 6
    nhead: int = 4
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.embed_dim)(x)

        class_token_embedding = self.param("class_token", nn.initializers.normal(), (1, 1, self.embed_dim))

        class_token = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))

        x = jnp.concatenate([class_token, x], axis=1)
        
        b, l, d = x.shape

        for _ in range(self.num_layer):
            x = TransformerEncoderLayerwithRelativePE(nhead=self.nhead, dropout_rate=self.dropout_rate)(x, training=training).reshape(b, l, d)

        x = x[:, 0, :]
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)

        return x

if __name__ == "__main__":

    m = T4HSCwithRoPE()

    x = jnp.ones((16, 10, 404))
    params = m.init({"params": jax.random.key(0), "dropout": jax.random.key(1)}, x, training=True)

    y = m.apply(params, x, training=True, rngs={"dropout": jax.random.key(0)})
    print(m.tabulate({"params": jax.random.key(0), "dropout": jax.random.key(1)}, jnp.ones((1, 100, 404)), True, compute_flops=True, compute_vjp_flops=False))
    print(jax.tree_util.tree_map(jnp.shape, params))


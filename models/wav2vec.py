import jax
import flax
import flax.linen as nn
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

class TransformerEncoderLayer(nn.Module):
    nhead: int = 4
    dropout_rate: float = 0.0

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

class Wav2Vec(nn.Module):

    num_layer: int = 6
    nhead: int = 4
    embed_dim: int = 256


    def random_masking(self, x, mask_ratio, mask_rng):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = jax.random.normal(mask_rng, (N, L))
        
        # sort noise for each sample
        ids_shuffle = jnp.argsort(noise, axis=-1)  # ascend: small is keep, large is remove
        ids_restore = jnp.argsort(ids_shuffle, axis=-1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked =  x[jnp.arange(x.shape[0]).reshape(-1, 1), ids_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones([N, L])
        mask = jax.lax.dynamic_update_slice(mask, jnp.zeros((N, len_keep)), (0, 0))
        # unshuffle to get the binary mask
        mask = mask[jnp.arange(mask.shape[0]).reshape(-1, 1), ids_restore]

        return x_masked, mask, ids_restore

    
    @nn.compact
    def __call__(self, x, mask_ratio):
        x_org = x
        x = nn.Dense(self.embed_dim)(x)

        position_encoding = sinusoidal_position_encoding(x.shape[-2] + 1, x.shape[-1])

        x = x + position_encoding[1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio, self.make_rng("mask"))

        class_token_embedding = self.param("class_token", nn.initializers.normal(), (1, 1, self.embed_dim)) + position_encoding[:1, :]
        class_tokens = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))
        x = jnp.concatenate([class_tokens, x], axis=1)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(nhead=self.nhead)(x, training=False)

        x = nn.LayerNorm()(x)

        x = nn.Dense(self.embed_dim)(x)


        mask_token_embedding = self.param("mask_token", nn.initializers.normal(), (1, 1, self.embed_dim))
        mask_tokens = jnp.tile(mask_token_embedding, (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1))

        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = x_[jnp.arange(x_.shape[0]).reshape(-1, 1), ids_restore]
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)

        x = x + position_encoding

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(nhead=self.nhead)(x, training=False)

        x = nn.LayerNorm()(x)

        x = nn.Dense(self.embed_dim)(x)

        y = x[:, 1:, :]

        mean = jnp.mean(x_org, axis=-1, keepdims=True)
        var = jnp.var(x_org, axis=-1, keepdims=True)
        target = (x_org - mean) / (var + 1.e-6) ** 0.5

        loss = (y - target) ** 2
        loss = jnp.mean(loss, axis=-1)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)

        return loss, y, mask

if __name__ == "__main__":
    m = Wav2Vec()

    x = jnp.ones((16, 10, 256))

    params = m.init({"params": jax.random.key(0), "mask": jax.random.key(2)}, x, mask_ratio=0.4)
    print(params)

    #y = m.apply(params, x, mask_ratio=0.4, rngs={"dropout": jax.random.key(1), "mask": jax.random.key(2)})
    loss, _, _ = m.apply(params, x, mask_ratio=0.4, rngs={"mask": jax.random.key(2)})

    #print(m.tabulate({"params": jax.random.key(0), "mask": jax.random.key(1)}, jnp.ones((1, 100, 256)), 0.4, compute_flops=True, compute_vjp_flops=False))

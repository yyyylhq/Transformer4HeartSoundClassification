import jax
import jax.numpy as jnp

# 设定随机种子以便复现结果
N = 3
L = 10
D = 7
len_keep = 6
key = jax.random.PRNGKey(0)

# 生成一个形状为 (3, 3) 的随机数组
x = jax.random.normal(jax.random.PRNGKey(0), (N, L, D))

print(x)

noise = jax.random.normal(key, (N, L))

ids_shuffle = jnp.argsort(noise, axis=-1)
ids_restore = jnp.argsort(ids_shuffle, axis=-1)

ids_keep = ids_shuffle[:, :len_keep]

x_masked =  x[jnp.arange(x.shape[0]).reshape(-1, 1), ids_keep]
#x_masked = jnp.array([x[i, ids_keep[i], ...] for i in range(x.shape[0])])
print(x_masked.shape)

mask = jnp.ones([N, L])
mask = jax.lax.dynamic_update_slice(mask, jnp.zeros((N, len_keep)), (0, 0))
print(mask)
print(ids_restore)

mask = mask[jnp.arange(mask.shape[0]).reshape(-1, 1), ids_restore]
#mask = mask[jnp.arange(mask.shape[0]), ids_restore]
print(mask)



mask_token_embedding = jax.random.normal(key, (1, 1, 256))
print(mask_token_embedding.shape)
mask_tokens = jnp.tile(mask_token_embedding, (3, 4, 1))
print(mask_tokens.shape)

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


position_encoding = sinusoidal_position_encoding(100 + 1, 256)

print(position_encoding.shape)

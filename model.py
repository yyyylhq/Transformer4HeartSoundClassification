import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

def sinusoidal_encoding(length, channels):
    position = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, channels, 2) * -(np.log(10000.0) / channels))
    pos_enc = np.zeros((length, channels))

    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)

    return jnp.array(pos_enc)



class TransformerEncoderLayer(nn.Module):
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.MultiHeadAttention(num_heads=self.nhead, qkv_features=self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(self.dim_feedfoward)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.LayerNorm()(x)

        return x

"""
class TransformerEncoderLayerwithRoPE(nn.Module):
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):

        x = nn.MultiHeadAttention(num_heads=self.nhead, qkv_features=self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(self.dim_feedfoward)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.LayerNorm()(x)

        return x
"""


class TransformerEncoder(nn.Module):
    num_layer: int = 4
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5


    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.d_model)(x)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedfoward=self.dim_feedfoward, dropout_rate=self.dropout_rate)(x, training=training)

        return x

class TransformerEncoderwithSinPE(nn.Module):
    num_layer: int = 4
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5
    class_token = jax.random.normal(jax.random.PRNGKey(0), (1, 1, d_model))

    @nn.compact
    def __call__(self, x, training=True):

        x = x + sinusoidal_encoding(x.shape[-2], x.shape[-1])

        class_tokens = jnp.tile(self.class_token, (1, 1, self.d_model))

        x = jnp.concatenate([class_tokens, x], axis=1)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedfoward=self.dim_feedfoward, dropout_rate=self.dropout_rate)(x, training=training)

        return x

"""
class TransformerEncoderwithRoPE(nn.Module):
    num_layer: int = 4
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.d_model)(x)

        for _ in range(self.num_layer):
            x = TransformerEncoderLayerwithRoPE(d_model=self.d_model, nhead=self.nhead, dim_feedfoward=self.dim_feedfoward, dropout_rate=self.dropout_rate)(x, training=training)

        return x
"""

class T4HSC(nn.Module):
    num_classes: int = 2
    hidden_dim: int = 2048

    num_layer: int = 4
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5

    #class_token_embedding = jax.random.normal(jax.random.PRNGKey(0), (1, 1, d_model))

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.d_model)(x)

        x = x + sinusoidal_encoding(x.shape[-2], x.shape[-1])

        class_token_embedding = self.param('class_token', nn.initializers.normal(), (1, 1, self.d_model))

        class_token = jnp.tile(class_token_embedding, (x.shape[0], 1, 1))
        x = jnp.concatenate([class_token, x], axis=1)


        x = TransformerEncoder(num_layer=self.num_layer, d_model=self.d_model, nhead=self.nhead, dim_feedfoward=self.dim_feedfoward, dropout_rate=self.dropout_rate)(x, training)

        x = x[:, 0, :]
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)

        return x

if __name__ == "__main__":

    m = T4HSC()
    #m = TransformerEncoder()
    x = jnp.ones((16, 10, 404))
    params = m.init({"params": jax.random.key(0), "dropout": jax.random.key(1)}, x, training=True)
    print(*params)

    y = m.apply(params, x, training=False, rngs={'dropout': jax.random.key(0)})
    print(y.shape)
    #print(m.tabulate({'params': jax.random.key(0), 'dropout': jax.random.key(1)}, jnp.ones((1, 100, 404)), False, compute_flops=True, compute_vjp_flops=False))
    print(jax.tree_util.tree_map(jnp.shape, params))


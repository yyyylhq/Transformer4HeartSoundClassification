import jax
import flax.linen as nn
import jax.numpy as jnp

class TransformerEncoderLayer(nn.Module):
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.d_model)(x)
        x = nn.relu(x)
        x = nn.MultiHeadAttention(num_heads=self.nhead, qkv_features=self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.LayerNorm()(x)

        x = nn.Dense(self.dim_feedfoward)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(self.d_model)(x)
        x = x + nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        x = nn.LayerNorm()(x)

        #x = attention(x)
        #x = x + dropout0(x)
        #x = norm0(x)

        #x = x + dropout2(linear2(dropout1(activation(linear1(x)))))
        #x = norm1(x)

        return x

class TransformerEncoder(nn.Module):
    num_layer: int = 4
    d_model: int = 512
    nhead: int = 4
    dim_feedfoward: int = 2048
    dropout_rate: float = 0.5


    @nn.compact
    def __call__(self, x, train=True):
        layer_list = []
        for _ in range(self.num_layer):
            layer_list.append(TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedfoward=self.dim_feedfoward, dropout_rate=self.dropout_rate))

        for i in range(self.num_layer):
            x = layer_list[i](x, train=train)

        return x

if __name__ == "__main__":
    m = TransformerEncoder()
    x = jnp.ones((1, 10, 404))
    params = m.init({'params': jax.random.key(0), 'dropout': jax.random.key(1)}, x)

    for i in list(params):
        for j in list(params[i]):
            for k in list(params[i][j]):
                for l in list(params[i][j][k]):
                    print(i, j, k, l)

    x = jnp.ones((8, 10, 404))
    out = m.apply(params, x, train=True, rngs={'dropout': jax.random.key(0)})
    #print(m.tabulate({'params': jax.random.key(0), 'dropout': jax.random.key(1)}, jnp.ones((1, 100, 404)), compute_flops=True, compute_vjp_flops=False))

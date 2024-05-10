import jax
import flax
import optax
import functools
import jax.numpy as jnp

from clu import metrics
from flax import struct
from flax.training import train_state

from models import model

@struct.dataclass
class HeartSoundClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")

class HeartSoundClassificationTrainState(train_state.TrainState):
    metrics: HeartSoundClassificationMetrics

@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(rng, learning_rate):
    m = model.T4HSC()

    params = m.init(rng, jnp.ones((1, 100, 404)), True)["params"]
    tx = optax.adam(learning_rate)

    return HeartSoundClassificationTrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=tx,
        metrics=HeartSoundClassificationMetrics.empty()
    )

@functools.partial(jax.pmap, axis_name="ensemble")
def train_step(state, batch, dropout_rng):

    def loss_fn():
        logits = state.apply_fn({"params": state.params}, batch["data"], True, rngs=dropout_rng)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["label"]
        )

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn()

    return state.apply_gardients(grads=grads)



def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds)


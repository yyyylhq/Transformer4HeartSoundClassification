import jax
import optax
import model
import hsdataset
from clu import metrics
from flax.training import train_state
from flax import struct
from jax import numpy as jnp

@struct.dataclass
class HeartSoundClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")

class HeartSoundClassificationTrainState(train_state.TrainState):
    metrics: HeartSoundClassificationMetrics

def create_train_state(m, rng, learning_rate, momentum):
    params = m.init(rng, jnp.empty([1, 100, 404]))["params"]

    tx = optax.sgd(learning_rate, momentum)

    return train_state.TrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=tx,
        metrics=HeartSoundClassificationMetrics.empty()
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["data"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["label"]
        ).mean()

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state

@jax.jit
def compute_metrics(state, batch):
    logits = state.apply_fn({"params": state.params}, batch["data"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["label"]
    ).mean()

    metric_updates = state.metrics.single_from_model_output(
        logits=logits,
        labels=batch["label"],
        loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state

def train(args):

    #test_ds = hsdataset.HeartSoundDataset("./datasets/HS-PCCC2016/data", "./datasets/HS-PCCC2016/test/test.csv")


    m = model.TransformerEncoderwithSinPE()

    rng_params = jax.random.key(args.params_rng)
    rng_dropout = jax.random.key(args.dropout_rng)
    rng = {"params": rng_params, "dropout": rng_dropout}

    train_state = create_train_state(m=m, rng=rng, learning_rate=args.learning_rate, momentum=args.momentum)

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }



    for e in range(args.epochs):
        pass



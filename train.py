import jax
import optax
import model
import numpy as np
from clu import metrics
from flax.training import train_state
from flax import struct
from jax import numpy as jnp
#from hsdataset import HeartSoundDataset, HeartSoundDatasetVote
#from torch.utils.data import DataLoader

@struct.dataclass
class HeartSoundClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")

class HeartSoundClassificationTrainState(train_state.TrainState):
    metrics: HeartSoundClassificationMetrics

def create_train_state(m, rng, learning_rate, momentum):
    params = m.init(rng, jnp.empty([1, 100, 404]), True)["params"]

    tx = optax.sgd(learning_rate, momentum)

    return HeartSoundClassificationTrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=tx,
        metrics=HeartSoundClassificationMetrics.empty()
    )

@jax.jit
def train_step(state, batch, dropout_rng):

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["data"], True, rngs=dropout_rng)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["label"]
        ).mean()
        print(loss)

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state

@jax.jit
def compute_metrics(state, batch):
    logits = state.apply_fn({"params": state.params}, batch["data"], False)
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


    m = model.T4HSC()

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


    """
    train_dataset = HeartSoundDataset("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/train/{args.tf}/train.csv")
    test_dataset = HeartSoundDataset("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/test/{args.tf}/test.csv")
    test_vote_dataset = HeartSoundDatasetVote("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/test/{args.tf}/test.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    train_dataset = np.load(f"../datasets/PCCD/ten_folds/train/k{args.tf}/train_data.npy")
    train_dataset_label = np.load(f"../datasets/PCCD/ten_folds/train/k{args.tf}/train_label.npy")
    """
    train_dataset = np.load(f"../datasets/PCCD/ten_folds/train/k0/train_data.npy")
    train_dataset_label = np.load(f"../datasets/PCCD/ten_folds/train/k0/train_label.npy")


    for e in range(args.epochs):
        print(f"Epoch {e} start...")

        len = train_dataset.shape[0] // args.batch_size
        print(len)
        for i in range(len):
            batch = {}
            batch["data"] = jnp.array(train_dataset[i * args.batch_size : i * args.batch_size + args.batch_size])
            batch["label"] = jnp.array(train_dataset_label[i * args.batch_size : i * args.batch_size + args.batch_size].reshape(-1))
            print(batch["data"].shape)
            print(batch["label"].shape)
            print(batch["data"].dtype)
            print(batch["label"].dtype)

            rng_dropout, _ = jax.random.split(rng_dropout)
            train_state = train_step(train_state, batch, {"dropout": rng_dropout})
            train_state = compute_metrics(state=train_state, batch=batch)

            for metric,value in train_state.metrics.compute().items(): # compute metrics
                metrics_history[f'train_{metric}'].append(value) # record metrics


        
        print(f"loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        train_state = train_state.replace(metrics=train_state.metrics.empty())


        exit()



import jax
import time
import optax
import logging
import numpy as np

from clu import metrics
from flax.training import train_state
from flax import struct
from jax import numpy as jnp

import get_dataset
import dataloader
from models import model


@struct.dataclass
class HeartSoundClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")

class HeartSoundClassificationTrainState(train_state.TrainState):
    metrics: HeartSoundClassificationMetrics

def create_train_state(m, rng, learning_rate):
    params = m.init(rng, jnp.empty([1, 100, 256]), True)["params"]

    tx = optax.adam(learning_rate)

    return HeartSoundClassificationTrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=tx,
        metrics=HeartSoundClassificationMetrics.empty()
    )

@jax.jit
def train_step(state, batch, rngs):

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["data"], True, rngs=rngs)
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

    file_handler = logging.FileHandler(f"./log/log_{args.tf}.txt")
    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s: %(message)s"))
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(asctime)s: %(message)s"))

    logger = logging.getLogger(f"log_{args.tf}")
    logger.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    m = model.T4HSCwithRoPE(num_layer=6)

    rng_params = jax.random.key(args.params_rng)
    rng_dropout = jax.random.key(args.dropout_rng)

    rng = {"params": rng_params, "dropout": rng_dropout}

    train_state = create_train_state(m=m, rng=rng, learning_rate=args.learning_rate)

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }


    train_dataset = get_dataset.get_dataset("../datasets/PCCD/data/npy", "../datasets/PCCD/ten_folds/train/k0/train.csv")
    test_dataset = get_dataset.get_dataset("../datasets/PCCD/data/npy", "../datasets/PCCD/ten_folds/test/k0/test.csv")


    train_dataset_loader = dataloader.JAXDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataset_loader = dataloader.JAXDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    train_start_time = time.time()
    for e in range(args.epochs):
        e_start_time = time.time()
        logger.info("--------------------")
        logger.info(f"Epoch {e + 1}/{args.epochs} start...")

        i = 0
        for b in train_dataset_loader:
            batch = {
                "data": b[0],
                "label": b[1]
            }

            rng_dropout, _ = jax.random.split(rng_dropout)
            train_state = train_step(train_state, batch, rngs={"dropout": rng_dropout})
            train_state = compute_metrics(state=train_state, batch=batch)

            if (i + 1) % 10 == 0:
                logger.debug(f"Batch {i + 1}/{len(train_dataset_loader)} finished.")

            #logger.debug(f"Batch {i + 1}/{len(train_dataset_loader)} finished.")

            i += 1


        for metric,value in train_state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics

        test_metrics = HeartSoundClassificationMetrics.empty()
        for b in test_dataset_loader:
            batch = {
                "data": b[0],
                "label": b[1]
            }

            logits = train_state.apply_fn({"params": train_state.params}, batch["data"], False)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=batch["label"]
            ).mean()

            metric_updates = test_metrics.single_from_model_output(
                logits=logits,
                labels=batch["label"],
                loss=loss
            )
            test_metrics = test_metrics.merge(metric_updates)


        for metric,value in test_metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        logger.info(f"Loss: {(metrics_history['train_loss'][-1]):>6f}, accuracy: {(metrics_history['train_accuracy'][-1] * 100):>6f}.")
        logger.info(f"Loss: {(metrics_history['test_loss'][-1]):>6f}, accuracy: {(metrics_history['test_accuracy'][-1] * 100):>6f}.")
        logger.info(f"Epoch {e + 1} end, time: {(time.time() - e_start_time):.2f}s.")

        train_state = train_state.replace(metrics=train_state.metrics.empty())

    logger.info(f"Train end, time: {(time.time() - train_start_time):.2f}s.")

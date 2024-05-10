import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.model = "T4HSC"
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = 'imagenet2012:5.*.*'

    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.shuffle_buffer_size = 16 * 128
    config.prefetch = 10

    config.epochs = 100.0
    config.batch_size = 128
    config.learning_rate = 0.0001
    config.log_every_steps = 100


    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config


def metrics():
  return [
      "train_loss",
      "eval_loss",
      "train_accuracy",
      "eval_accuracy",
      "steps_per_second",
      "train_learning_rate"
  ]

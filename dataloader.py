from typing import Sequence
import jax.numpy as jnp
import jax.random as jrand

class BaseDataLoader:

    """
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        pass
    """

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

def EpochIterator(
    data,
    batch_size: int,
    indices: Sequence[int]
):
    for i in range(0, len(indices), batch_size):
        idx = indices[i : i + batch_size]
        yield data[idx]

class JAXDataLoader(BaseDataLoader):

    def __init__(
        self, 
        dataset,
        batch_size: int = 1,  # batch size
        shuffle: bool = True,  # if true, dataloader shuffles before sampling each batch
        drop_last: bool = False,
        shuffle_PRNGKey: int = 0
    ):

        self.shuffle_PRNGKey = jrand.PRNGKey(shuffle_PRNGKey)
        self.dataset = dataset

        self.indices = jnp.arange(len(dataset))

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if  len(self.indices) % self.batch_size == 0:
            return len(self.indices) // self.batch_size

        return len(self.indices) // self.batch_size + int(not self.drop_last)

    def __iter__(self):
        indices = jrand.permutation(self.next_key(), self.indices).__array__() if self.shuffle else self.indices

        if self.drop_last:
            indices = indices[:len(self.indices) - (len(self.indices) % self.batch_size)]

        return EpochIterator(self.dataset, self.batch_size, indices)


    def next_key(self):
        self.key, subkey = jrand.split(self.shuffle_PRNGKey)

        return subkey


if __name__ == "__main__":
    pass

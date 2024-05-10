import jax
import numpy as np

class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __gettiem__(self):
        raise NotImplementedError

class ArrayDataset(Dataset):

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        assert x.shape[0] == y.shape[0], "All arrays must have the same dimension."

        self.arrays = (x, y)

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        return jax.tree_util.tree_map(lambda x: x[index], self.arrays)

if __name__ == "__main__":
    pass

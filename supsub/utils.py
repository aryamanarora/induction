import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import typing
import os

T = typing.TypeVar('T', bound='ModelHelper')
PathComp = typing.Union[Path, str]
"""a type for path-compatible objects"""

def get_data_dir() -> Path:
    """get a directory to store cached data"""
    return ensure_dir(Path(__file__).resolve().parent.parent / 'data')

def get_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class ModelHelper:
    """a mixin class to add basic functionality to torch modules"""

    def save_to_file(self, fpath):
        """save the model to file"""
        torch.save(self, fpath)

    @classmethod
    def load_from_file(cls: typing.Type[T], fpath) -> T:
        net = torch.load(fpath, map_location=torch.device('cpu'))
        assert isinstance(net, cls)
        return net

    def get_device(self) -> str:
        """get the device of this model"""
        for i in self.parameters():
            return i.device
        raise RuntimeError('can not guess device without params')

def torch_as_npy(x: torch.Tensor) -> np.ndarray:
    return x.data.cpu().numpy()

def init_torch_seed(seed):
    """initialize torch seed from a seed compatible with numpy SeedSequence"""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    torch.manual_seed(rng.integers(np.iinfo(np.int64).max))

def as_path(p: PathComp) -> Path:
    if isinstance(p, Path):
        return p
    return Path(p)

def ensure_dir(p: PathComp) -> Path:
    """ensure that a directory exists
    :return: ``p`` as Path
    """
    p = as_path(p)
    if not p.is_dir():
        os.makedirs(p)
    return p

def assign_zero_grad(x: torch.Tensor) -> torch.Tensor:
    """set gradient associated with a tensor to zero; similar to
    :meth:`torch.nn.Module.zero_grad`

    :return: x
    """
    if x.grad is not None:
        x.grad.detach_().zero_()
    return x

def iter_dataset_infinity(dataset: Dataset, batch_size: int, drop_last: bool):
    """iterate over a pytroch dataset infinitely, with shuffle"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=drop_last, num_workers=2)
    loader_iter = iter(loader)
    while True:
        try:
            yield next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)

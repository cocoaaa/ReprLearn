from .types_ import *
from torch import nn
from abc import abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

class BaseVAE(LightningModule):
    
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """Given an input image x, returns the reconstructed image"""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass


class BaseGAN(LightningModule):

    def __init__(self) -> None:
        super().__init__()

    def decode(self, input: Tensor) -> Any:
        """Pass through the generator: batch of z --> batch of x_gen's"""
        raise NotImplementedError

    def discriminate(self, input:Tensor, label:Tensor) -> Any:
        """Pass through the discriminator: """
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    # @abstractmethod
    # def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
    #     pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass




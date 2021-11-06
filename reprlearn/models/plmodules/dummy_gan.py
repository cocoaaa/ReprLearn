from .types_ import *
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .base import BaseGAN
from reprlearn.models.convnet import conv_blocks, deconv_blocks
from reprlearn.models.resnet import ResNet
from reprlearn.models.resnet_deconv import ResNetDecoder


class DummyGenerator(nn.Module):
    """Receives a random noise vector and generate a binary representation of an integer
    of length input_len
    """

    def __init__(self, dim_z: int, dim_x: int):
        super().__init__()
        self.fc = nn.Linear(dim_z, dim_x)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        return self.act_fn(self.fc(x))

class DummyGenerator2(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DummyDiscr(nn.Module):

    def __init__(self, dim_x, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        dim_out = 1 if self.n_classes == 2 else self.n_classes
        self.fc = nn.Linear(dim_x, dim_out)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        return self.act_fn(self.fc(x))

class DummyDiscr2(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class DummyGAN(BaseGAN):

    def __init__(self, *,
                decoder: nn.Module,
                discr: nn.Module,
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 latent_dim: int,
                 hidden_dims: List,
                 learning_rate: float,
                 act_fn: Callable = nn.LeakyReLU(),
                 out_fn: Callable = nn.Tanh(),
                 size_average: bool = False,
    ) -> None:
        self.decoder = decoder
        self.discr = discr
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.act_fn = act_fn
        self.out_fn = out_fn
        self.size_average = size_average


    @property
    def name(self):
        bn = 'DummyGAN'
        return f'{bn}'

    def input_dim(self):
        return np.prod(self.dims)

    def on_fit_start(self, *args, **kwargs):
        print(f"{self.__class__.__name__} is called")

    def forward(self, x: Tensor, **kwargs) -> Dict[str,Tensor]:
        """TODO: FIX
        Forward-prop input batch x through encoder and decoder.
        Returns a dict of parameters for the q_z ("mu","log_var") and the reconstruction "recon"

        Returns
        -------
        out : Dict[str, Tensor] of latent parameters "mu", "log_var" and
           data-likelihood model distribution's parameter mu_x (as "recon")
        """
        pass

    def loss_function(self, out, target, mode:str,
                      **kwargs) -> dict:
        """TODO: fix
        Computes the VAE loss function from a mini-batch of pred and target
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Args
        ----
        out : Dict[Tensor] returned by forward method

        mode : (str) one of "train", "val", "test"
        kwargs : eg. has a key "kld_weight" to multiply the (negative) kl-divergence

        Returns
        -------
        loss_dict : Dict of per-datapoint loss terms for the batch
             keys are `recon_loss`, `kld`, `loss`
        """
        pass

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor: #TODO
        """todo: fix
        Samples from the latent space and return samples from approximate data space:
        z ---> mu of P_x|z

        Args
        ----
        num_samples: (Int) Number of samples
        current_device: (Int) Device to run the model; it must be same as
            where model weight tensors are located (ie. self.device)

        Returns
        -------
        samples : batch of reconstructions from sampled latent codes
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            z = z.to(current_device)
            samples = self.decode(z)
            return samples

    def training_step(self, batch, batch_idx): #todo
        """Implements one mini-batch iteration:
         batch input -> pass through model (enc, reparam, dec) -> loss (ie. computational graph)
        """
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="train")

        # -- log each component of the loss
        self.log('train/recon_loss', loss_dict["recon_loss"])
        self.log('train/kld', loss_dict["kld"])
        self.log('train/vae_loss', loss_dict["loss"])
        self.log('train/loss', loss_dict["loss"])

        return {'loss': loss_dict["loss"]}

    def validation_step(self, batch, batch_ids): #TODO
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="val")

        self.log('val/recon_loss', loss_dict["recon_loss"])
        self.log('val/kld', loss_dict["kld"])
        self.log('val/vae_loss', loss_dict["loss"])
        self.log('val/loss', loss_dict["loss"])

        if (self.current_epoch+1) % 10 == 0 and (self.trainer.batch_idx+1) % 300 == 0:
            print(f"Ep: {self.trainer.current_epoch+1}, batch: {self.trainer.batch_idx+1}, loss: {loss_dict['loss']}")

        return {"val_loss": loss_dict["loss"]}

    def test_step(self, batch, batch_idx): #TODO
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="test")
        self.log('test/loss', loss_dict["loss"], prog_bar=True, logger=True)

        return {"test_loss": loss_dict["loss"]}

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        opt_discr = optim.Adam(self.discr.parameters(), lr=self.learning_rate)

        # lr_scheduler = {
        #     'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                       mode='min',
        #                                                       patience=10,
        #                                                       verbose=True),
        #     'monitor': 'val_loss',
        #     'name': "train/lr/Adam",
        # }
        return [opt_gen, opt_discr], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--in_shape', nargs=3,  type=int, required=True)
        parser.add_argument('--enc_type', type=str, default="conv")
        parser.add_argument('--dec_type', type=str, default="conv")
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--hidden_dims', nargs="+", type=int) #None as default
        parser.add_argument('--act_fn', type=str, default="leaky_relu", help="Choose relu or leaky_relu (default)") #todo: proper set up in main function needed via utils.py's get_act_fn function
        parser.add_argument('--out_fn', type=str, default="tanh", help="Output function applied at the output layer of the decoding process. Default: tanh")
        parser.add_argument('--kld_weight', type=float, default=1.0)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

        return parser
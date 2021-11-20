from typing import List, Callable, Union, Optional, TypeVar, Tuple, Dict
Tensor = TypeVar('torch.tensor')
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from pprint import pprint
from .base import BaseVAE

class IWAE(BaseVAE):
    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int,int,int]],
                 latent_dim: int,
                 hidden_dims: List,
                 n_samples: int,
                 learning_rate: float,
                 act_fn: Callable= nn.LeakyReLU(),
                 size_average: bool = False,
                 **kwargs) -> None:
        """
        :param in_shape: model(x)'s input x's shape w/o batch dimension
         in order of (c, h, w). Note no batch dimension.
        :param latent_dim:
        :param hidden_dims:
        :param n_samples: number of latent codes to draw from q^{(n}) corresponding
        to the variational distribution of nth datapoint.
            Note. If `n_samples==1`, IWAE is the same model as Vanilla VAE.
        :param act_fn: Default is LeakyReLU()
        :param learning_rate: initial learning rate. Default: 1e-3.
        :param size_average: bool; whether to average the recon_loss across the pixel dimension.
            Default: False
        :param kwargs: will be part of self.hparams
            Eg. batch_size, kld_weight
        """
        super().__init__()
        self.dims = in_shape
        self.n_channels, self.in_h, self.in_w = in_shape
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.act_fn = act_fn
        self.learning_rate = learning_rate
        self.size_average = size_average
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]  # , 512] # after encoder network, the input's h,w will be 1/(2^5) = 1/32 sized.
        self.hidden_dims = hidden_dims

        # Save kwargs to tensorboard's hparams
        self.save_hyperparameters()

        # Compute last feature map's height, width
        self.n_layers = len(self.hidden_dims)
        self.last_h, self.last_w =int(self.in_h/2**self.n_layers), int(self.in_w/2**self.n_layers)

        # Build Encoder
        modules = []
        in_c = self.n_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    self.act_fn)
            )
            in_c = h_dim

        self.encoder = nn.Sequential(*modules)
        self.len_flatten = hidden_dims[-1] * self.last_h * self.last_w
        self.fc_mu = nn.Linear(self.len_flatten, latent_dim)
        self.fc_var = nn.Linear(self.len_flatten, latent_dim)


        # Build Decoder
        modules = []
        self.fc_latent2decoder = nn.Linear(self.latent_dim, self.len_flatten)
        rev_hidden_dims = hidden_dims[::-1]

        for i in range(len(rev_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev_hidden_dims[i],
                                       rev_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(rev_hidden_dims[i + 1]),
                    self.act_fn)
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(rev_hidden_dims[-1],
                                               self.n_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.n_channels),
                            self.act_fn,

                            nn.Conv2d(self.n_channels, self.n_channels,
                                      kernel_size=3, stride=1, padding= 1),
                            nn.Tanh()) #todo: sigmoid? maybe Tanh is better given we normalize inputs by mean and std

    @property
    def name(self):
        return f'IWAE_{self.n_samples}'

    def input_dim(self):
        return np.prod(self.dims)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode_one_z(self, z: Tensor) -> Tensor:
        """
        Maps a batch of latent codes onto the image space, when each row contains
        a single latent code corresponding to the rowth input datapoint

        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.fc_latent2decoder(z)
        result = result.view(-1, self.hidden_dims[-1], self.last_h, self.last_w)
        result = self.decoder(result);  # print(result.shape)
        result = self.final_layer(result);  # print(result.shape)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) (BS, num_samples, latent_dim)
        :return: (Tensor) (BS, num_samples, C, H, W)
        """
        bs = z.shape[0]
        assert self.n_samples == z.shape[1] #todo: remove it after testing
        mu_x_pred = torch.zeros((bs, self.n_samples, *self.dims), device=z.device, dtype=z.dtype)

        for (i, z_samples) in enumerate(z):
            # z_samples: (num_samples, latent_dim)
            out = self.decode_one_z(z_samples) # (n_samples, C,H,W)
            mu_x_pred[i] = out

        return mu_x_pred

    def rsample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Sample latent codes  from N(mu, var) by using the reparam. trick.

        :param mu: (Tensor) Mean of the latent Gaussian [B x latent_dim]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x latent_dim]
        :return: (Tensor) [BS, self.num_zs, self.latent_dim]
        """
        # Add `num_zs`'s dimension to mu and logvar
        bs = mu.shape[0]
        mu = mu[:,None]  #(BS, 1, latent_dim)
        logvar = logvar[:, None] #(BS, 1, latent_dim)
        std = torch.exp(0.5 * logvar)

        eps = torch.randn((bs, self.n_samples, self.latent_dim), device=mu.device, dtype=mu.dtype)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        mu, log_var = self.encode(x)
        z = self.rsample(mu, log_var)

        return  {"mu":mu, "log_var":log_var, "mu_x_pred":self.decode(z)}

    def loss_function(self, out, target, mode:str,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function from a mini-batch of pred and target
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param mode: (str) one of "train", "val", "test"
        :param kwargs: eg. has a key "kld_weight" to multiply the (negative) kl-divergence
        :return:
        """
        num_data = getattr(self.trainer.datamodule, f"n_{mode}") # Warning: this is set only after self.trainer runs its datamodule's "setup" method
        # print("mode, num_data: ", mode, num_data)
        mu, log_var, mu_x_pred  = out["mu"], out["log_var"], out["mu_x_pred"]
        try:
            kld_weight = kwargs['kld_weight']
        except KeyError:
            kld_weight = 1.0

        # Adds a dimension correpsonding to the `n_samples` dimension
        target = target[:, None]

        # Compute losses
        recon_loss = (mu_x_pred - target).pow(2).mean(dim=1).sum(dim=(-3, -2, -1)).mean(dim=0)
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_loss + kld_weight * kld

        # Estimates for per-datapoint (ie. image), computed as an average over mini-batch
        # TODO: Noisy gradient estimate of the (full-batch) gradient thus need to be multipled by num_datapoints N
        loss_dict = {'loss': loss,
                     'recon_loss': recon_loss,
                     'kld': kld}

        if self.current_epoch % 10 == 0 and self.trainer.batch_idx % 300 == 0:
            print(f"Ep: {self.current_epoch}, batch: {self.trainer.batch_idx}")
            pprint(loss_dict)
        return loss_dict #{'loss': loss, 'recon_loss': recon_loss, 'kld': kld}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        TODO:
        CHECK IF taking mean makes sense
        """
        mu_x_pred = self.forward(x)["mu_x_pred"] # (BS, num_samples, C, H, W)
        return mu_x_pred.mean(dim=1)

    def get_embedding(self, x: Tensor, **kwargs) -> List[Tensor]:
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.rsample(mu, log_var)
            return {"mu": mu, "log_var": log_var, "z": z}

    def training_step(self, batch, batch_idx):
        """
        Implements one mini-batch iteration:
         from batch intack -> pass through model -> return loss (ie. computational graph)
        """
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="train")
        # breakpoint()
        # Log using tensorboard logger
        # For scalar metrics, self.log will do
        self.log('train/loss', loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.acc(logits, y)
        # self.log('train/acc', self.acc, on_step=True, on_epoch=False)

        # Alternatively, construct a log_dict and add it to the return statement
        # log_dict = {'train_acc', self.acc(logits, y)}
        # breakpoint()
        # if self.trainer.current_epoch%100 == 0:
        #     print(f'Epoch {self.trainer.current_epoch} -- lr: {self.hparams.learning_rate}')
        return {'loss': loss_dict["loss"],
                'log': loss_dict}

    # def training_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train/acc_epoch', self.acc.compute())

    def validation_step(self, batch, batch_ids):
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="val")

        self.log('val/loss', loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Method 1-(1)accumulate metrics on validation batch
        # self.log('val/acc_step', self.val_acc.compute())

        # Preferred. Method 1-(2)
        # self.val_acc(logits, y)
        # self.log('val/acc', self.val_acc, on_step=True, on_epoch=True)

        return {"val_loss": loss_dict["loss"],
                'log': loss_dict}

    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train/acc_epoch', self.val.accu.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="test")

        self.log('test/loss', loss_dict["loss"], prog_bar=True, logger=True)
        # Preferred. Method 1-(2)
        # self.test_acc(logits, y)
        # self.log('test/acc', self.test_acc, on_epoch=True)

        return {"val_loss": loss_dict["loss"],
                'log': loss_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate"))

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        # override existing arguments with new ones, if exists
        if parent_parser is not None:
            parents = [parent_parser]
        else:
            parents = []

        parser = ArgumentParser(parents=parents, add_help=False, conflict_handler='resolve')
        # parser.add_argument('--in_shape', nargs="3",  type=int, default=[3,64,64])
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--hidden_dims', nargs="+", type=int) #None as default
        parser.add_argument('--n_samples', type=int, required=True)
        parser.add_argument('--act_fn', type=str, default="leaky_relu")
        parser.add_argument('-lr', '--learning_rate', type=float, default="1e-3")

        return parser
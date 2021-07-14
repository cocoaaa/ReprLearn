from .types_ import *
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from pprint import pprint
from .base import BaseVAE
from src.models.convnet import conv_blocks, deconv_blocks
from src.models.resnet import ResNet
from src.models.resnet_deconv import ResNetDecoder

class BetaVAE(BaseVAE):
    """
    Encode method:
     x -> encoder(x) # (BS, hidden_dim[-1], last_h, last_w)
       -> Flatten()
       -> FC(len_flatten, latent_dim)

    Decode method:
    z -> fc_latent2decoder + reshape 1d to 3d #(BS, hidden_dim[-1], last_h, last_w)
      -> out_fn: eg. nn.Tanh(), nn.Sigmoid()

    """
    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int,int,int]],
                 latent_dim: int,
                 hidden_dims: List,
                 learning_rate: float,
                 act_fn: Callable = nn.LeakyReLU(),
                 out_fn: Callable = nn.Tanh(),
                 size_average: bool = False,
                 kld_weight: float = 1.0,
                 enc_type: str = 'conv',
                 dec_type: str = 'conv',
                 ** kwargs) -> None:
        """
        :param in_shape: model(x)'s input x's shape w/o batch dimension
         in order of (c, h, w). Note no batch dimension.
        :param latent_dim:
        :param hidden_dims:
        :param act_fn:
        :param out_fn: output function at the last layer of the decoder network
        :param learning_rate: initial learning rate. Default: 1e-3.
        :param size_average: bool; whether to average the recon_loss across the pixel dimension.
            Default: False
        :param kld_weight : float
            weight balancing the recon_loss and kld; corresponds to "beta" in beta-VAE
            The scale of kld_weight is per input data point (, rather than per datapt's pixel, ie. dim_x)
        :param use_resnet: If true, implement encder and decoder with skip connections
        :param kwargs: will be part of self.hparams
            Eg. batch_size, kld_weight
        """
        super().__init__()
        self.dims = in_shape
        self.in_channels, self.in_h, self.in_w = in_shape
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.out_fn = out_fn
        self.learning_rate = learning_rate
        self.size_average = size_average
        self.kld_weight = kld_weight
        self.enc_type = enc_type
        self.dec_type = dec_type
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]#, 512] # after encoder network, the input's h,w will be 1/(2^5) = 1/32 sized.
        self.hidden_dims = hidden_dims
        self.example_input_array = torch.zeros((1, *in_shape), dtype = self.dtype)

        # Save kwargs to tensorboard's hparams
        self.save_hyperparameters()

        # Compute last feature map's height, width
        # In case of resnet, the second convlayer doesn't shrink the resolution (h,w)
        self.n_downsampling_layers = self.get_n_downsampling_layers()
        self.last_h = int(self.in_h/2**self.n_downsampling_layers)
        self.last_w = int(self.in_w/2**self.n_downsampling_layers)

        # Build Encoder
        if self.enc_type == 'conv':
            self.encoder = conv_blocks(self.in_channels,
                                       self.hidden_dims,
                                       has_bn=True,
                                       act_fn=act_fn)
        elif self.enc_type == 'resnet':
            self.encoder = ResNet(self.in_channels,
                                  self.hidden_dims,
                                  act_fn=act_fn)
        else:
            raise NotImplementedError("Currently supports convnet, resnet as encoder")

        self.len_flatten = self.hidden_dims[-1] * self.last_h * self.last_w
        self.fc_mu = nn.Linear(self.len_flatten, latent_dim)
        self.fc_var = nn.Linear(self.len_flatten, latent_dim)

        # Build Decoder
        self.fc_latent2flatten = nn.Linear(self.latent_dim, self.len_flatten)
        if self.enc_type == 'resnet':
            decoder_dims =[*hidden_dims[1:][::-1], self.in_channels]
        else:
            decoder_dims = [*hidden_dims[::-1], self.in_channels]

        if self.dec_type == 'conv':
            self.decoder = deconv_blocks(decoder_dims[0],
                                         decoder_dims[1:],
                                         has_bn=True,
                                         act_fn=act_fn)  # bs, (len_flatten,) -> ... -> bs, (n_channels, h,w)
        elif self.dec_type == 'resnet':
            self.decoder = ResNetDecoder(decoder_dims,
                                         act_fn=act_fn) #todo
        else:
            raise NotImplementedError("Currently supports convnet, resnet as decoder")


        self.out_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=3, stride=1, padding= 1),
            self.out_fn) #todo: sigmoid? maybe Tanh is better given we normalize inputs by mean and std

    @property
    def name(self):
        bn = 'BetaVAE'
        return f'{bn}-{self.enc_type}-{self.dec_type}-{self.kld_weight:.3f}'

    def input_dim(self):
        return np.prod(self.dims)

    def get_n_downsampling_layers(self):
        if self.enc_type == 'conv':
            return len(self.hidden_dims)
        elif self.enc_type == 'resnet':
            return len(self.hidden_dims) - 1
        else:
            raise NotImplementedError("Use a valid enc_type: 'conv', 'resnet'")

    def on_fit_start(self, *args, **kwargs):
        print(f"{self.__class__.__name__} is called")

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
        # Q: apply nonlinearity after fc?
        # A: (Jan 27, 2021) no, because we don't want to constrain the range of mu (or var)
        # with any implicit bias. In the original paper Kingma2014, they don't
        # apply nonlinearity either
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """ Maps the given latent code onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.fc_latent2flatten(z); #print(result.shape); breakpoint()
        result = result.view(-1, self.hidden_dims[-1], self.last_h, self.last_w) #; print(result.shape); breakpoint()
        result = self.decoder(result) #; print(result.shape); breakpoint()
        result = self.out_layer(result) #; print(result.shape); breakpoint()
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        # import ipdb; ipdb.set_trace()
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  {"mu":mu, "log_var":log_var, "recon":self.decode(z)}

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
        mu, log_var, recon  = out["mu"], out["log_var"], out["recon"]

        recon_loss = F.mse_loss(recon, target, reduction='mean', size_average=self.size_average) # see https://github.com/pytorch/examples/commit/963f7d1777cd20af3be30df40633356ba82a6b0c
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_loss + self.kld_weight * kld

        # Estimates for per-datapoint (ie. image), computed as an average over mini-batch
        # TODO: Noisy gradient estimate of the (full-batch) gradient thus need to be multipled by num_datapoints N
        loss_dict = {
           'recon_loss': recon_loss,
           'kld': kld,
           'loss': loss,
        }

        return loss_dict

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
        """

        return self.forward(x)["recon"]

    def get_embedding(self, x: Tensor, **kwargs) -> List[Tensor]:
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            return {"mu": mu, "log_var": log_var, "z": z}

    def training_step(self, batch, batch_idx):
        """
        Implements one mini-batch iteration:
         from batch intack -> pass through model -> return loss (ie. computational graph)
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

    def validation_step(self, batch, batch_ids):
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="val")
        self.log('val/loss', loss_dict["loss"])
        self.log('val/recon_loss', loss_dict["recon_loss"])
        self.log('val/kld', loss_dict["kld"])
        self.log('val/vae_loss', loss_dict["loss"])
        self.log('val/loss', loss_dict["loss"])

        if self.current_epoch % 10 == 0 and self.trainer.batch_idx % 300 == 0:
            print(f"Ep: {self.trainer.current_epoch}, batch: {self.trainer.batch_idx}, loss: {loss_dict['loss']}")

        return {"val_loss": loss_dict["loss"]}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss_dict = self.loss_function(out, x.detach().clone(), mode="test")
        self.log('test/loss', loss_dict["loss"], prog_bar=True, logger=True)

        return {"test_loss": loss_dict["loss"]}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              patience=10,
                                                              verbose=True),
            'monitor': 'val_loss',
            'name': "train/lr/Adam",
        }

        return [optimizer], [lr_scheduler]
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--in_shape', nargs=3,  type=int, required=True)
        parser.add_argument('--enc_type', type=str, default="conv")
        parser.add_argument('--dec_type', type=str, default="conv")
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--hidden_dims', nargs="+", type=int) #None as default
        parser.add_argument('--act_fn', type=str, default="leaky_relu") #todo: proper set up in main function needed via utils.py's get_act_fn function
        # parser.add_argument('--out_fn', type=str, default="tanh")
        parser.add_argument('--kld_weight', type=float, default=1.0)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)



        return parser
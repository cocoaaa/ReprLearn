from .types_ import *
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
import torchvision
from .base import BaseGAN

# GAN model
class GAN(BaseGAN):
    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 latent_dim: int,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 learning_rate: float,
                 niter_D_per_G: int,
    #         label_map: Optional[Dict] = None,
                 size_average: bool = False,
                 n_show: int = 16,  # number of images to log at checkpt iteration
                **kwargs,
    ):
        super().__init__()
        self.dims = in_shape
        self.in_channels, self.in_h, self.in_w = in_shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.niter_D_per_G = niter_D_per_G # "k" in Goodfellow2014, ie. update D k-times given G; then given the udpated D, update G once
        #         self.label_map = label_map or {"real": 1, "gen": 0}
        self.size_average = size_average
        self.generator = generator
        self.discr = discriminator

        self.automatic_optimization = False #take care of optimizers manually
        # Adam parameters
        self.b1 = kwargs.get('b1', None) or 0.99
        self.b2 = kwargs.get('b2', None) or 0.99

        self.example_input_array = torch.zeros((2, self.latent_dim),
                                               dtype=self.dtype)

        # Save kwargs to tensorboard's hparams
        self.save_hyperparameters()


@property
def name(self):
    bn = "GAN"
    return f'{bn}-{self.generator.name}-{self.discr.name}-dimz:{self.latent_dim}'


def input_dim(self):
    return np.prod(self.dims)


def on_fit_start(self, *args, **kwargs):
    print(f"{self.__class__.__name__} is called")


def forward(self, z: Tensor):
    return self.generator(z)


def adversarial_loss(self, score: Tensor, is_real: bool):
    return self.discr.compute_loss(score, is_real)


def loss_fn_D(self, batch: Tensor, mode: str): #todo: finish it
    # freeze G
    # unfreeze D
    x, _ = batch

    # Generator
    # 1. sample noise
    z = torch.randn(x.shape[0], self.latent_dim)
    z = z.type_as(x)
    # 2. pass through generator
    # continue here!!!
    self.x_gen = self(z)
    score = self.discr(self.x_gen)
    # maximize log(D(x_gen)), aka. minimize -log((D(x_gen))
    loss_G = self.discr.compute_loss(score, is_real=True)
    # alternatively, use the modified loss function as in Goodfellow14
    # loss_G = torch.log(nn.Sigmoid(score))

    # return loss_dict = {
    #     'loss_G': loss_G,
    #     'loss_D_real': loss_D_real,
    #     'loss_D_gen': loss_D_gen,
    #     'loss_D': loss_D
    # }
    pass


def loss_fn_G(self, batch: Tensor, mode: str):  # todo: finish it
    # freeze D
    # unfreeze G
    x, _ = batch

    # Generator
    # 1. sample noise
    z = torch.randn(x.shape[0], self.latent_dim)
    z = z.type_as(x)
    # 2. pass through generator
    self.x_gen = self(z)
    score = self.discr(self.x_gen)
    # maximize log(D(x_gen)), aka. minimize -log((D(x_gen))
    loss_G = self.discr.compute_loss(score, is_real=True)
    # alternatively, use the modified loss function as in Goodfellow14
    # loss_G = torch.log(nn.Sigmoid(score))

    # return loss_dict = {
    #     'loss_G': loss_G,
    # }
    pass

 
def sample(self,
           num_samples: int,
           current_device: int,
           **kwargs) -> Tensor:
    """Samples from the latent space and return samples from approximate data space:
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
        z = torch.randn((num_samples, self.latent_dim),
                        device=current_device)  # z = z.type_as(?)
        samples = self.generator(z)
        return samples


def sample_train_batch(self) -> [Tensor,Tensor]:
    """Return a random batch from the training dataloader
    In order to not affect the training step's dataloader,
    we make a deepcopy of a dataloader and take a first random batch from it
    """
    dl = deepcopy(self.train_dataloader())
    batch = next(iter(dl))
    return batch


def training_step(self, batch, batch_idx, optimizer_idx):
    """Implements one mini-batch iteration:
     batch input -> pass through model (enc, reparam, dec) -> loss (ie. computational graph)
    """
    imgs, _ = batch
    bs = len(imgs)
    opt_g, opt_d = self.optimizers(use_pl_optimizer=False)
    ######################
    # Update D, k times  #
    ######################
    # freeze G
    self.generator.training(False)
    self.discr.training(True)
    for k in self.niter_D_per_G:
        # sample noise
        z = torch.randn(bs, self.latent_dim)
        z = z.type_as(imgs)
        # pass through generator
        x_gen = self(z).detach() # detach so that we block gradient flowing to G
        # score for generated data
        score_gen = self.discr(x_gen)
        # Compute loss_D_gen
        loss_D_gen =self.discr.compute_loss(score_gen, is_real=False)

        # sample mini-batch from real dataset
        x_real = self.sample_train_batch()
        score_real = self.discr(x_real)
        # Compute loss_D_real
        loss_D_real = self.discr.compute_loss(score_real, is_real=True)

        # loss_D
        loss_D = 0.5 * (loss_D_real, loss_D_gen)

        # Update D
        opt_d.zero_grad()
        self.manual_backward(loss_D)
        opt_d.step()

    ######################
    #  Update G once     #
    ######################
    # freeze D
    self.discr.training(False)
    self.generator.training(True)

    # sample noise
    z = torch.randn(bs, self.latent_dim)
    z = z.type_as(imgs)

    # pass through generator
    x_gen = self(z)
    # get critic from the discriminator
    score_G_gen = self.discr(x_gen).detach() #detach important
    # objective for G: want score_G_gen to be high; fool discr
    loss_G = self.discr.compute_loss(x_gen, is_real=True) # True bc we want discr to be fooled

    # Update G
    opt_g.zero_grad()
    self.manual_backward(loss_G)
    opt_g.step()


    # show/log a couple generated images
    # todo: make this into a logger callback
    sample_x_gen = x_gen[:self.n_show]
    grid = torchvision.utils.make_grid(sample_x_gen)
    self.logger.experiment.add_image("generated_images", grid, self.trainer.current_epoch)  # todo: ep should be set properly


def configure_optimizers(self):
    lr = self.learning_rate
    b1 = self.b1
    b2 = self.b2

    opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
    opt_d = torch.optim.Adam(self.discr.parameters(), lr=lr, betas=(b1, b2))
    return opt_g, opt_d #no learning rate scheduler


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



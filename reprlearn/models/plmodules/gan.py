from .types_ import *
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from reprlearn.models.utils import inplace_freeze, inplace_unfreeze

# class GAN(BaseGAN):
class GAN(LightningModule):

    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int, int, int]],
                 latent_dim: int,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 learning_rate: float,
                 niter_D_per_G: int,
    #         label_map: Optional[Dict] = None,
                 size_average: bool = False,
                 log_every:int = 10, #log generated images every this epoch
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

        # Optimizers: take care of them manually
        # self.automatic_optimization = False
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/5108#issuecomment-768252625

        self.b1 = kwargs.get('b1', None) or 0.50#.99 #todo: find good values
        self.b2 = kwargs.get('b2', None) or 0.50#.99

        self.example_input_array = torch.zeros((2, self.latent_dim),
                                               dtype=self.dtype,
                                               device='cuda' #todo: remove
                                               )

        # Save kwargs to tensorboard's hparams
        self.save_hyperparameters()

        # don't need to log these as model hparams
        self.n_show = n_show
        self.log_every = log_every


    @property
    def automatic_optimization(self) -> bool:
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/5108#issuecomment-768252625
        return False

    @property
    def name(self) -> str:
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

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
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
            z = torch.randn((num_samples, self.latent_dim), device=current_device)  # z = z.type_as(?)
            samples = self.generator(z)
            return samples

    def sample_train_batch(self) -> [Tensor, Tensor]:
        """Return a random batch from the training dataloader
        In order to not affect the training step's dataloader,
        we make a deepcopy of a dataloader and take a first random batch from it
        """
        dl = deepcopy(self.train_dataloader())
        batch = next(iter(dl))
        return batch[0].to(self.device), batch[1].to(self.device)

    def push_through_D_and_G(self, batch) -> Dict[str,Tensor]:
        """Compute losses of G and D, using the input batch (if applicable).
        Not suitable for training_step (due to no optimizer incorporate),
        but useful for val/test_step

        Returns a dict of the losses:
        - loss_G: current G's generative power to make images that fools current D
            - NB1: also involves current D's state
            - NB2: no input images in the validation dataset is used to validate G
        - loss_D_real: current D's power to predict real images as real
            - NB: this is the only loss that needs the input images in the validation dataset
        - loss_D_gen: current D's power to predict the generated images as 'fake'
            - NB: this loss also depends on how good current G is
        - loss_D: averaged loss that is signaled to current D

         """
        # with torch.no_grad(): -- handled by pl
        # let's just make sure
        # print('Debug: push_through_D_and_G...')
        # print('\tis G in train mode: (should be false) ', self.generator.training)
        # print('\tis D in train mode: (should be false)', self.generator.training)

        imgs, _ = batch
        bs = len(imgs)
        ######################
        # Pass through D     #
        ######################
        # Compute loss_D_real
        # pass-through D to check how well it predicts x_real 'real'
        x_real, _ = batch
        score_real = self.discr(x_real)
        # Compute loss_D_real
        loss_D_real = self.discr.compute_loss(score_real, is_real=True)

        # Compute loss_D_gen
        # first, sample noise
        z = torch.randn(bs, self.latent_dim)
        z = z.type_as(imgs)
        # pass through generator
        x_gen = self(z).detach()  # detach shouldn't be needed as long as PL is setting the context of 'torch.no_grad' properly?
        # score for generated data
        score_gen = self.discr(x_gen)
        # Compute loss_D_gen
        loss_D_gen = self.discr.compute_loss(score_gen, is_real=False)
        # loss_D
        loss_D = 0.5 * (loss_D_real + loss_D_gen)

        ######################
        #  Pass-through G    #
        ######################
        # sample noise
        z = torch.randn(bs, self.latent_dim)
        z = z.type_as(imgs)
        # pass through generator
        x_gen = self(z)
        # get critic from the discriminator
        # objective for G: want score_G_gen to be high; fool discr
        score_G_gen = self.discr(x_gen).detach() #detach important for training
        t = self.discr.label_map['real']
        target_label = t * torch.ones_like(score_G_gen)
        loss_G = self.discr.loss_fn(score_G_gen, target_label)

        return {
            "loss": loss_G, #TODO
            "loss_G": loss_G,
            "loss_D_real": loss_D_real,
            "loss_D_gen": loss_D_gen,
            "loss_D": loss_D
        }

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Implements one mini-batch iteration:
         batch input -> pass through model (enc, reparam, dec) -> loss (ie. computational graph)
        """
        imgs, _ = batch
        bs = len(imgs)
        # opt_g, opt_d = self.optimizers(use_pl_optimizer=False)
        opt_g, opt_d = self.optimizers()

        ######################
        # Update Discr, k times  #
        ######################
        # print('Starting k updates for D...')
        # freeze generator
        inplace_freeze(self.generator)
        for k in range(self.niter_D_per_G):
            # sample noise
            z = torch.randn(bs, self.latent_dim)
            z = z.type_as(imgs)
            # print('z device: ', z.device)

            # pass through generator
            x_gen = self(z).detach() # detach so that we block gradient flowing to G
            # print('x_gen device: ', x_gen.device)

            # score for generated data
            score_gen = self.discr(x_gen)
            # Compute loss_D_gen
            loss_D_gen =self.discr.compute_loss(score_gen, is_real=False)

            # sample mini-batch from real dataset
            x_real, _ = self.sample_train_batch()
            score_real = self.discr(x_real)
            # Compute loss_D_real
            loss_D_real = self.discr.compute_loss(score_real, is_real=True)

            # loss_D
            loss_D = 0.5 * (loss_D_real + loss_D_gen)

            # Update D
            opt_d.zero_grad()
            self.manual_backward(loss_D, opt_d)
            opt_d.step()
        # unfreeze back the generator
        inplace_unfreeze(self.generator)

        ######################
        #  Update G once     #
        ######################
        # print('Starting one step update of G...')
        # freeze D
        inplace_freeze(self.discr)
        # sample noise
        z = torch.randn((bs, self.latent_dim))
        z = z.type_as(imgs)
        # print('z device: ', z.device)
        # pass through generator
        x_gen = self.generator(z)
        score_G_gen = self.discr(x_gen)
        loss_G = self.discr.compute_loss(score_G_gen, is_real=True)  # True bc we want D to be fooled
        # Update G
        opt_g.zero_grad()
        self.manual_backward(loss_G, opt_g)
        opt_g.step()

        # unfreeze back the discriminator
        inplace_unfreeze(self.discr)

        ####################################
        #  log each component of the loss  #
        ####################################
        self.log('train/loss_G', loss_G)
        self.log('train/loss_D', loss_D)
        self.log('train/loss_D_gen', loss_D_gen)
        self.log('train/loss_D_real', loss_D_real)

        # Debug
        print('train/loss_G', loss_G)
        print('train/loss_D', loss_D)
        print('train/loss_D_gen', loss_D_gen)
        print('train/loss_D_real', loss_D_real)
        breakpoint()

        # show/log a couple generated images
        # todo: make this into a logger callback
        if (self.trainer.current_epoch+1) % self.log_every == 0:
            print(f'Epoch {self.trainer.current_epoch+1} ...')
            print(f'-- loss_G: {loss_G.item():.3f}') # detectability of x_gen, lower means harder to distinguish x_gen from real better)
            print(f'-- loss_D_gen:  {loss_D_gen.item():.3f}')
            print(f'-- loss_D_real:  {loss_D_real.item():.3f}')

            sample_x_gen = x_gen[:self.n_show]
            grid = torchvision.utils.make_grid(sample_x_gen)
            self.logger.experiment.add_image("generated_images", grid, self.trainer.current_epoch)  # todo: ep should be set properly

        return {"loss": loss_G}  # TODO

    def validation_step(self, batch, batch_ids) -> Dict[str,Tensor]:
        """Evaluate losses on validation dataset (if applicable)
        Returns a dict of the losses:
        - loss_G: current G's generative power to make images that fools current D
            - NB1: also involves current D's state
            - NB2: no input images in the validation dataset is used to validate G
        - loss_D_real: current D's power to predict real images as real
            - NB: this is the only loss that needs the input images in the validation dataset
        - loss_D_gen: current D's power to predict the generated images as 'fake'
            - NB: this loss also depends on how good current G is
        - loss_D: averaged loss that is signaled to current D

         """
        # with torch.no_grad(): -- handled by pl
        # let's just make sure
        # print('Debug: Validating...')
        # print('\tis G in train mode: ', self.generator.training)
        # print('\tis D in train mode: ', self.generator.training)

        loss_dict = self.push_through_D_and_G(batch)

        # -- log each component of the loss
        self.log('val/loss_G', loss_dict['loss_G'])
        self.log('val/loss_D', loss_dict['loss_D'])
        self.log('val/loss_D_gen', loss_dict['loss_D_gen'])
        self.log('val/loss_D_real', loss_dict['loss_D_real'])

        return {"val_loss": loss_dict['loss_G']} #TODO

    def test_step(self, batch, batch_idx):
        # with torch.no_grad(): -- handled by pl
        # let's just make sure
        # print('Debug: Testing...')
        # print('\tis G in train mode: ', self.generator.training)
        # print('\tis D in train mode: ', self.generator.training)

        loss_dict = self.push_through_D_and_G(batch)

        # -- log each component of the loss
        self.log('test/loss_G', loss_dict['loss_G'])
        self.log('test/loss_D', loss_dict['loss_D'])
        self.log('test/loss_D_gen', loss_dict['loss_D_gen'])
        self.log('test/loss_D_real', loss_dict['loss_D_real'])

        return {"test_loss": loss_dict['loss_G']}  # TODO

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



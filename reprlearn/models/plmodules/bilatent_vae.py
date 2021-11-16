from typing import List, Callable, Union, Optional, TypeVar, Tuple, Dict
Tensor = TypeVar('torch.tensor')
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics import Accuracy

from .base import BaseVAE
from .utils import compute_kld

from reprlearn.models.convnet import conv_blocks, deconv_blocks
from reprlearn.models.resnet import ResNet
from reprlearn.models.resnet_deconv import ResNetDecoder


class BiVAE(BaseVAE):
    def __init__(self, *,
                 in_shape: Union[torch.Size, Tuple[int,int,int]],
                 n_styles: int,

                 latent_dim: int,
                 hidden_dims: Optional[List[int]],
                 adversary_dims: Optional[List[int]],
                 n_samples: int,
                 learning_rate: float,
                 act_fn: Callable= nn.LeakyReLU(),
                 out_fn: Callable = nn.Tanh(),

                 size_average: bool = False,

                 is_contrasive: bool = True,
                 kld_weight: float = 1.0,
                 adv_loss_weight: float = 1.0,

                 enc_type: str = 'conv',
                 dec_type: str = 'conv',
                 **kwargs) -> None:
        """
        VAE with extra adversarial loss from a style discriminator to enforce the
        information from original data to be encoded into two independent subspaces
        of the latent space, \mathcal{Z_c} and \mathcal{Z_s} aka. Bi-latent VAE
        TODO: how about Bilinear VAE

        :param in_shape: model(x)'s input x's shape w/o batch dimension,
        in order of (c, h, w). Note no batch dimension.
        :param latent_dim:
        :param hidden_dims:
        :param n_samples: number of latent codes to draw from q^{(n}) corresponding
        to the variational distribution of nth datapoint.
            Note. If `n_samples==1`, our model is the same model as Vanilla VAE.
        :param act_fn: Default is LeakyReLU()
        :param learning_rate: initial learning rate. Default: 1e-3.
        :param size_average: bool; whether to average the recon_loss across the
        pixel dimension. Default: False
        :param is_contrasive bool; True to use both adversarial losses from the
        content and style codes
            If False, use only the loss from the style code's classification prediction a
            s the adversarial loss
        :param kld_weight (float); Beta in BetaVAE that is a relative weight of
        the kld vs. recon-loss
            vae_loss = recon_loss + kld_weight * kld
        :param adv_loss_weight (float); Weight btw vae_loss and adv_loss
            loss = vae_loss + adv_loss_weight * adv_loss
        :param enc_type (str); One of ['resnet', 'conv']
        :param dec_type(str); One of ['resnet', 'conv']
        :param kwargs: will be part of self.hparams
            Eg. batch_size, kld_weight
        """
        super().__init__()
        # About input x
        self.dims = in_shape
        self.in_channels, self.in_h, self.in_w = in_shape
        # About label y
        self.n_styles = n_styles # num of styles from which the adversary to predict
        # About model configs
        self.latent_dim = latent_dim
        self.content_dim = int(self.latent_dim/2)
        self.style_dim = self.latent_dim - self.content_dim
        self.n_samples = n_samples
        self.act_fn = act_fn
        self.out_fn = out_fn
        self.learning_rate = learning_rate
        self.size_average = size_average
        self.hidden_dims = hidden_dims or [32, 64, 128, 256, 512]
        self.adversary_dims = adversary_dims or  [32, 32, 32]
        # Loss
        self.is_contrasive = is_contrasive
        self.kld_weight = kld_weight
        self.adv_loss_weight = adv_loss_weight
        # Encoder, Decoder type
        self.enc_type = enc_type
        self.dec_type = dec_type
        # Save kwargs to tensorboard's hparams
        self.save_hyperparameters()

        # Compute last feature map's height, width
        # In case of resnet, the second convlayer doesn't shrink the resolution (h,w)
        self.n_downsampling_layers = self.get_n_downsampling_layers()
        self.last_h = int(self.in_h / 2 ** self.n_downsampling_layers)
        self.last_w = int(self.in_w / 2 ** self.n_downsampling_layers)

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
        self.fc_flatten2qparams = nn.Linear(self.len_flatten, 2*self.content_dim+2*self.style_dim) # mu_qc, std_qc, mu_qs, std_qs (both c, s have the same dim)

        # Build Decoder
        # modules = []
        # self.fc_latent2decoder = nn.Linear(self.latent_dim, self.len_flatten)
        # rev_hidden_dims = self.hidden_dims[::-1]
        #
        # for i in range(len(rev_hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(rev_hidden_dims[i],
        #                                rev_hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(rev_hidden_dims[i + 1]),
        #             self.act_fn)
        #     )
        #
        # self.decoder = nn.Sequential(*modules)
        #
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(rev_hidden_dims[-1],
        #                                        self.n_channels,
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(self.n_channels),
        #                     self.act_fn,
        #
        #                     nn.Conv2d(self.n_channels, self.n_channels,
        #                               kernel_size=3, stride=1, padding= 1),
        #                     nn.Tanh()) #todo: sigmoid? maybe Tanh is better given we normalize inputs by mean and std
        self.fc_latent2flatten = nn.Linear(self.latent_dim, self.len_flatten)

        if self.enc_type == 'resnet':
            decoder_dims = [*hidden_dims[1:][::-1], self.in_channels]
        else:
            decoder_dims = [*hidden_dims[::-1], self.in_channels]

        if self.dec_type == 'conv':
            self.decoder = deconv_blocks(decoder_dims[0],
                                         decoder_dims[1:],
                                         has_bn=True,
                                         act_fn=act_fn)  # bs, (len_flatten,) -> ... -> bs, (n_channels, h,w)
        elif self.dec_type == 'resnet':
            self.decoder = ResNetDecoder(decoder_dims,
                                         act_fn=act_fn)  # todo
        else:
            raise NotImplementedError("Currently supports convnet, resnet as decoder")

        self.out_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=3, stride=1, padding=1),
            self.out_fn)  # todo: sigmoid? maybe Tanh is better given we normalize
                          # inputs by mean and std

        # Build style classifier:
        # Given a content or style code, predict its style label
        # zc or zs --> scores (a vector of len = n_classes)
        _adv_dims = [self.content_dim, *self.adversary_dims, self.n_styles]
        adv_layers = []
        for num_in, num_out in zip(_adv_dims, _adv_dims[1:]):
            adv_layers.append(nn.Sequential(nn.Linear(num_in, num_out), self.act_fn))
        self.adversary = nn.Sequential(*adv_layers)

        # Add the accuracy metric for the style-classification based on the style code
        self.train_style_acc = Accuracy()
        self.val_style_acc = Accuracy()
        self.test_style_acc = Accuracy()

        # todo:Add confusion matrix metric as multi-class classification metric
        # for style-prediction subtask
        # self.train_cm =

    @property
    def name(self):
        bn = "BiVAE-C" if self.is_contrasive else "BiVAE"
        return f'{bn}-{self.enc_type}-{self.dec_type}-{self.kld_weight:.1f}-{self.adv_loss_weight:.1f}-{self.n_samples}'

    def input_dim(self):
        return np.prod(self.dims)

    def get_n_downsampling_layers(self):
        if self.enc_type == 'conv':
            return len(self.hidden_dims)
        elif self.enc_type == 'resnet':
            return len(self.hidden_dims) - 1
        else:
            raise NotImplementedError("Use a valid enc_type: 'conv', 'resnet'")

    def encode(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Dict[std,Tensor]) Dict of parameters for variational distributions q_content and q_style
         dict_q_params = {
            "mu_qc": mu_qc,
            "logvar_qc": logvar_qc,
            "mu_qs": mu_qs,
            "logvar_qs": logvar_qs
        }
        """
        out = self.encoder(input)
        out = torch.flatten(out, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        q_params = self.fc_flatten2qparams(out) #(bs, 2*content_dim + 2*style_dim]
        mu_qc = q_params[:, 0:self.content_dim]
        logvar_qc = q_params[:, self.content_dim:self.content_dim*2]
        mu_qs = q_params[:, self.content_dim*2:self.content_dim*2+self.style_dim]
        logvar_qs = q_params[:, self.content_dim*2+self.style_dim:]

        dict_q_params = {
            "mu_qc": mu_qc,
            "logvar_qc": logvar_qc,
            "mu_qs": mu_qs,
            "logvar_qs": logvar_qs
        }
        return dict_q_params

    def rsample(self, dict_q_params: Dict[str, Tensor]) -> Dict[str,Tensor]:
        """
        Sample latent codes  from N(mu, var) by using the reparam. trick.

        :param dict_q_params: output of the encoder network
        :return: dict_z_samples (Dict[str, Tensor])
            keys: 'c', 's'
            value of dict_zsample['c']: samples of content codes; [BS, self.latent_dim]
            - same for key='s'
        """
        mu_qc = dict_q_params["mu_qc"]  #(BS, self.content_dim)
        logvar_qc = dict_q_params["logvar_qc"] #(BS, self.content_dim)
        std_qc = logvar_qc.exp()
        mu_qs = dict_q_params["mu_qs"] #(BS, self.style_dim)
        logvar_qs = dict_q_params["logvar_qs"] #(BS, self.style_dim)
        std_qs = logvar_qs.exp()

        # Expand mu's and logvar's to 3-dim tensor to handle >1
        # sampling of z's from each q(z[n])
        mu_qc = mu_qc[:, None]  # (BS, 1, self.content_dim)
        std_qc = std_qc[:, None] # (BS, 1, self.content_dim)
        mu_qs = mu_qs[:, None]  # (BS, 1, self.content_dim)
        std_qs = std_qs[:, None]  # (BS, 1, self.content_dim)

        # Reparam. trick
        # eps_c = torch.randn_like(mu_qc)
        bs = len(mu_qc)
        eps_c = torch.randn((bs, self.n_samples, self.content_dim),
                            dtype=mu_qc.dtype,
                            device=mu_qc.device)
        c_samples = eps_c * std_qc + mu_qc

        eps_s = torch.randn((bs, self.n_samples, self.style_dim),
                            dtype=mu_qs.dtype,
                            device=mu_qs.device)
        s_samples = eps_s * std_qs + mu_qs

        # todo: not sure if this is a good idea
        # but at least now, this will keep the code backward compatible
        # take the average of the z's sampled from q(z[n])
        c_samples = c_samples.mean(1)
        s_samples = s_samples.mean(1)
        # print("c_samples: ", c_samples.shape)
        # print("s_samples: ", s_samples.shape)
        # breakpoint()

        dict_z_samples = {"c": c_samples,"s": s_samples}
        return dict_z_samples

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps a batch of latent codes onto the image space, when each row contains
        a single latent code corresponding to the rowth input datapoint

        :param z: (Tensor) [B, latent_dim]
        :return: (Tensor) [B, C, H, W]
        """
        out = self.fc_latent2flatten(z) # latent_dim -> len_flatten; 1dim tensor
        out = out.view(-1, self.hidden_dims[-1], self.last_h, self.last_w) # back to a mini-batch of 3dim tensors
        out = self.decoder(out); #print(out.shape)
        out = self.out_layer(out); #print(out.shape)

        return out

    def create_labels(self, z):
        """
        Create proper target_labels for c and s

        :param z:
        :return:
        """
        pass

    def discriminate(self, z: Tensor) -> Tensor:
        """
        - Divide z into c and s
        -
        :param z:
        :return: y_pred: style label prediction (BS, )
        """
        pass

    def combine_content_style(self, dict_z: Dict[str, Tensor]) -> Tensor:
        """
        Combine a mini-batch of content codes and a mini-batch of style codes
        to get a "full" sample of z that can be fed into the decoder
        :param dict_z: Dict with keys "c", "s". dict_z["c"] returns a mini-batch of content codes.
        :return: a mini-batch of z = [zc, zs] vectors
        """
        c = dict_z["c"] # (BS, content_dim)
        s = dict_z["s"] # (BS, style_dim)
        assert len(c) == len(s), "Number of content and style codes must be the same"
        return torch.cat([c, s], dim=1)

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        Full forwardpass of VAE: x -> enc -> rsample(z's) -> dec
        :param x: mini-batch of inputs (BS, *in_shape)
        :param kwargs:
        :return: Dict[str,Tensor] with keys
            "mu_qc", "logvar_qc", "mu_qs", "logvar_qs", "c", "s", mu_x_pred"
        """
        dict_q_params = self.encode(x)
        dict_z_samples = self.rsample(dict_q_params)
        z = self.combine_content_style(dict_z_samples) # (BS, self.latent)
        mu_x_pred = self.decode(z)
        out_dict = {**dict_q_params, **dict_z_samples, "mu_x_pred":mu_x_pred}
        return  out_dict

    # ------------------------------------------------------------------------
    # Methods for adversary
    # ------------------------------------------------------------------------
    def partition_z(self, z: Tensor) -> Dict[str, Tensor]:
        """
        Reverse operation of `combine_content_style`.
        Given a (batch of) latent code z, divide it into content and style codes.
        :param z:
        :return: dict_z
        """
        dict_z = {
            "c": z[:, :self.content_dim],
            "s": z[:,  self.content_dim:]
        }
        return dict_z

    def predict_d(self, z_partition):
        """
        Use the style classifer to predict the style label given either content or style code.
        :param z_partition:  (BS, self.content_dim), same as (BS, style_dim)
        :return: y_scores: predicted styles  (BS, n_styles)
        """
        d_scores = self.adversary(z_partition) #(BS, n_styles)
        return d_scores

    def compute_loss_c(self, c:torch.Tensor) -> Tensor:
        """
        Using the current adversary, compute the prediction loss of style
        given the content codes.
        - Set the target to be a uniform dist. over classes. ie (BS, n_styles)
        with values = 1/n_styles

        :param c:
        :return: loss_c (torch.float32)
        """
        bs = len(c)
        # target = torch.ones((bs, self.n_styles), device=c.device)
        # target /= self.n_styles # TODO: possible to not create this as a tensor, as it has all the same value ie. 1/self.n_styles
        scores = self.predict_d(c); #print("score_c: ", scores.shape) #(bs, n_styles)
        log_probs = nn.LogSoftmax(dim=1)(scores) #(bs, n_styles)
        loss_c = - log_probs.mean(dim=1) # same as: log_probs.sum(dim=1) / self.n_styles
        loss_c = loss_c.mean(dim=0) # adversarial loss per content code

        return loss_c

    def compute_loss_s(self, s:torch.Tensor, target_d) -> Tensor:
        """

        :param s: style code; (bs, style_dim)
        :param target_d: target style index (ie. one-hot style target); (bs,)
        :return: loss_s (torch.float32)
        """
        scores = self.predict_d(s)
        loss_s = nn.CrossEntropyLoss(reduction='mean')(scores, target_d) # estimated loss computed as averaged loss (over batch)
        return loss_s

    def loss_function(self,
                      out_dict,
                      batch,
                      mode:str,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function from a mini-batch of pred and target
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        :param out_dict: output of the full forward pass
        :param target: Tuple[ Tensor, Tensor]. mini-batch of inputs and labels
        :param mode: (str) one of "train", "val", "test"
        :param kwargs:
            eg. has a key "kld_weight" to multiply the (negative) kl-divergence
        :return: loss_dict
        """
        # Uppack the batch into a batch of img, content_labels, style_labels
        # target_x = batch['img'].detach().clone()
        # label_c = batch['digit']  # digit/content label (int); not used currently
        # label_s = batch['color'].detach().clone()  # color/style label (int); used for adversarial loss_s
        target_x, label_c, label_s = self.trainer.datamodule.unpack(batch)
        target_x = target_x.detach().clone() # imgs
        label_s = label_s.detach().clone() # d

        # qparams
        mu_qc, logvar_qc = out_dict["mu_qc"], out_dict["logvar_qc"]
        mu_qs, logvar_qs = out_dict["mu_qs"], out_dict["logvar_qs"]
        # sampled latent codes
        c, s, = out_dict["c"], out_dict["s"]
        # output of decoder
        mu_x_pred = out_dict["mu_x_pred"]

        # Monitor kld of each latent subspace to see how the content/style
        # latent's KLD's changes individually
        with torch.no_grad():
            kld_c = compute_kld(mu_qc, logvar_qc)
            kld_s = compute_kld(mu_qs, logvar_qs)

        # Combine mu_qc and mu_qs. Same for logvars. Defines q_z for each datapt
        mu_z = self.combine_content_style({"c": mu_qc, "s": mu_qs})
        logvar_z = self.combine_content_style({"c": logvar_qc, "s": logvar_qs})

        # Compute losses
        recon_loss = F.mse_loss(mu_x_pred, target_x, reduction='mean', size_average=self.size_average) # see https://github.com/pytorch/examples/commit/963f7d1777cd20af3be30df40633356ba82a6b0c
        # kld = torch.mean(-0.5 * torch.sum(1 + logvar_z - mu_z ** 2 - logvar_z.exp(), dim = 1), dim = 0)
        kld = compute_kld(mu_z, logvar_z)
        vae_loss = recon_loss + self.kld_weight * kld

        # Compute adversarial loss
        # adv_loss_s = self.compute_loss_s(s, label_s) #loss from "positives"
        score_s = self.predict_d(s); #print("score_s: ", score_s.shape) #(bs, n_styles)
        adv_loss_s = nn.CrossEntropyLoss(reduction='mean')(score_s, label_s)  # estimated loss computed as averaged loss (over batch)
        if self.is_contrasive:
            adv_loss_c = self.compute_loss_c(c)  # loss from "negatives"
            adv_loss = adv_loss_c + adv_loss_s
        else:
            adv_loss = adv_loss_s

        # Finally, full loss
        # Estimates for per-datapoint (ie. image), computed as an average over mini-batch
        # TODO: Noisy gradient estimate of the (full-batch) gradient thus need to be multipled by num_datapoints N
        loss = vae_loss + self.adv_loss_weight * adv_loss

        loss_dict = {
            "kld_c": kld_c,
            "kld_s": kld_s,
            'recon_loss': recon_loss,
            'kld': kld,
            'vae_loss': vae_loss,
            "score_s": score_s,# (BS, n_styles); needed to compute the style accuracy at `training_step`
            'adv_loss_s': adv_loss_s,
            "adv_loss": adv_loss,
            "loss": loss
        }
        if self.is_contrasive:
            loss_dict["adv_loss_c"] = adv_loss_c

        if self.current_epoch % 10 == 0 and self.trainer.batch_idx % 300 == 0:
            print(f"Ep: {self.current_epoch}, batch: {self.trainer.batch_idx}")
            # pprint(loss_dict)

        return loss_dict

    def generate(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """Generate images from marginal p_z:
        Given model's marginal dist. of Cr.v and Sr.v. are both
        multivariate standard Gaussian (mu=zero vector, cov_mtx is identity mtx), samples from this marginal latent space

        and return the corresponding image space map
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # Given model's marginal dist. of Cr.v and Sr.v. are both
        # multivariate standard Gaussian (mu=zero vector, cov_mtx is identity mtx)
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            z = z.to(current_device)
            mu_x_pred = self.decode(z)
            return mu_x_pred

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        TODO:
        CHECK IF taking mean makes sense
        """
        mu_x_pred = self.forward(x)["mu_x_pred"] # (BS, C, H, W)
        return mu_x_pred

    def get_embedding(self, x: Tensor, **kwargs) -> List[Tensor]:
        self.eval()
        with torch.no_grad():
            dict_q_params = self.encode(x)
            dict_z = self.rsample(dict_q_params)
            c = dict_z["c"]  # (BS, content_dim)
            s = dict_z["s"]  # (BS, style_dim)
            return {"c": c, "s": s}

    def training_step(self, batch, batch_idx):
        """
        Implements one forward-pass (aka. "iteration"):
        x -> model(x) -> loss or loss_dict
        `loss` is the last node of the model's computational graph,
        ie. starting node of backprop.
        """
        # x = batch['img']
        # # label_c = batch['digit'] # digit/content label (int) -- currently not used
        # label_s = batch['color'] # color/style label (int) -- used for adversarial loss_s

        x, label_c, label_s = self.trainer.datamodule.unpack(batch) # "unpack" must be implemented in the DataModule class -- Jan 22, 2021
        out_dict = self(x)
        loss_dict = self.loss_function(out_dict, batch, mode="train")
        # breakpoint()

        # Log using tensorboard logger
        #-- for scalar metrics, self.log will do
        self.log('train/loss', loss_dict["loss"]) # Default: on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #-- log each component of the loss
        self.log('train/vae_loss', loss_dict["vae_loss"])
        self.log('train/recon_loss', loss_dict["recon_loss"])
        self.log('train/kld', loss_dict["kld"])
        self.log('train/kld_c', loss_dict["kld_c"])
        self.log('train/kld_s', loss_dict["kld_s"])

        self.log('train/adv_loss', loss_dict["adv_loss"])
        self.log('train/adv_loss_s', loss_dict["adv_loss_s"])
        if self.is_contrasive:
            self.log('train/adv_loss_c', loss_dict["adv_loss_c"])

        #-- update and log the style_acc metric
        score_s = loss_dict.pop("score_s").detach().clone() # we don't want to compute metric on the loss computational graph
        self.train_style_acc(score_s, label_s)
        self.log('train/style_acc', self.train_style_acc) # Note: we pass in the Metric object, rather than the value tensor

        return {'loss': loss_dict["loss"]}


    def validation_step(self, batch, batch_ids):
        # x = batch['img']
        # # label_c = batch['digit']  # digit/content label (int) -- currently not used
        # label_s = batch['color']  # color/style label (int) -- used for adversarial loss_s
        x, label_c, label_s = self.trainer.datamodule.unpack(batch) # "unpack" must be implemented in the DataModule class -- Jan 22, 2021
        out_dict = self(x)
        loss_dict = self.loss_function(out_dict, batch, mode="val")

        # Log the validation loss
        self.log('val/loss', loss_dict["loss"])  # Default: on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # -- log each component of the loss
        self.log('val/vae_loss', loss_dict["vae_loss"])
        self.log('val/recon_loss', loss_dict["recon_loss"])
        self.log('val/kld', loss_dict["kld"])
        self.log('val/kld_c', loss_dict["kld_c"])
        self.log('val/kld_s', loss_dict["kld_s"])

        self.log('val/adv_loss', loss_dict["adv_loss"])
        self.log('val/adv_loss_s', loss_dict["adv_loss_s"])
        if self.is_contrasive:
            self.log('val/adv_loss_c', loss_dict["adv_loss_c"])

        # Update and log val_style_acc metric
        score_s = loss_dict.pop('score_s').detach().clone()
        self.val_style_acc(score_s, label_s)
        self.log('val/style_acc', self.val_style_acc)
        # print(self.trainer.callback_metrics.keys())
        # ['val_loss', 'train/loss', 'train/vae_loss', 'train/recon_loss', 'train/kld', 'train/kld_c', 'train/kld_s', 'train/adv_loss', 'train/adv_loss_s', 'train/adv_loss_c', 'train/style_acc']
        return {"val_loss": loss_dict["loss"]}


    def test_step(self, batch, batch_idx):
        # x = batch['img']
        # # label_c = batch['digit']  # digit/content label (int) -- currently not used
        # label_s = batch['color']  # color/style label (int) -- used for adversarial loss_s
        x, label_c, label_s = self.trainer.datamodule.unpack(batch) # "unpack" must be implemented in the DataModule class -- Jan 22, 2021
        out_dict = self(x)
        loss_dict = self.loss_function(out_dict, x.detach().clone(), mode="test")

        self.log('test/loss', loss_dict["loss"])

        # Update and log test_style_acc metric
        score_s = loss_dict.pop('score_s').detach().clone()
        self.test_style_acc(score_s, label_s)
        self.log('test/style_acc', self.test_style_acc, on_step=False, on_epoch=True)

        return {"test_loss": loss_dict["loss"]}

    def configure_optimizers(self):
        #TODO: ADD optimizer for discriminator
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
        # parser.add_argument('--in_shape', nargs="3",  type=int, default=[3,64,64])
        # Required
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--n_styles', type=int, required=True)
        # Recommended
        # -- Model architecture
        parser.add_argument('--hidden_dims', nargs="+", type=int) #None as default
        parser.add_argument('--adv_dims', dest="adversary_dims", nargs="+", type=int) #None as default
        parser.add_argument('--act_fn', type=str, default="leaky_relu")
        parser.add_argument('--enc_type', type=str, default="conv")
        parser.add_argument('--dec_type', type=str, default="conv")

        # -- Loss function
        parser.add_argument('--kld_weight', type=float, default=1.0)
        parser.add_argument('--adv_weight', dest="adv_loss_weight", type=float, default=1.0)
        # Add boolean argument switches for contrasive loss
        # src: https://stackoverflow.com/a/31347222
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--is_contrasive', dest='is_contrasive', action='store_true')
        group.add_argument('--not_contrasive', dest='is_contrasive', action='store_false')
        parser.set_defaults(is_contrasive=True)

        # -- Optimizer(s)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
        return parser


class Encoder(nn.Module):
    """
    Input of (BS, C, H, W); Outputs (BS, *out_shape)
    - Each layer is implemented as a conv-block, conv2D -> (BN2D) -> out_fn (eg. ReLU)

    Args:
        -
    """
    def __init__(self):
        super().__init__()
        # self.convs =


class Decoder(nn.Module):
    pass


class Adversary(nn.Module):
    pass


class FCAdversary(Adversary):
    pass


class CNNAdversary(Adversary):
    pass
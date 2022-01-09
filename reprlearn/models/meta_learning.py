from .types_ import *
from copy import deepcopy
from collections import Counter
from .convnet import conv_blocks
from .utils import init_zero_grad, get_grad_ckpt, has_same_values, zero_grad, inplace_freeze, inplace_unfreeze
from reprlearn.data.datasets.kshot_dataset import KShotImageDataset
from reprlearn.data.samplers.kshot_sampler import KShotSampler
from reprlearn.visualize.utils import show_timgs
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from IPython.core.debugger import set_trace as breakpoint


class MAML(nn.Module):

    def __init__(self,
                 in_shape: List[int],
                 k_shot: int,
                 n_way: int,
                 device: Union[str, torch.device],
                 net: Optional[nn.Module]=None,
                 task_loss_fn: Optional[Callable] = None,
                 **kwargs,
                 ):
        """
        MAML's meta-generator and task-learners are all of the same model-class
        (ie. same parametric neural network).
        They have the same dimensions and only values are different.

        simple feed-forward + avaerage network
        - convnet as feature-extractor of an image
        - add each input image's features at `forward`

        Computational graph
        -------------------
        input: support_set = {(x_1,y_1), ..., (x_n, y_n)}
        self.convnet: works on each image x_i
        self.fc: works on output (flatten to 1dim) of self.convent(x_i) and outputs a
          vector of length `dim_h`
        self.h2phi: expands a latent code (h) for the model parameters for this task-learner
          to a vector of length `dim_phi` (ie., the actual model parameter's dimension)
          This is theta_g in ICML 2019 tutorial.


        Args
        ----
        in_shape: List[int]
          shape of an input datapt, (h,w,nc)
        k_shot : int
            number of images per class in support/query of a task-set
        n_way : int
            Number of classes or target variable for each task
        device
            'cpu', 'cuda'
        net : nn.Module
            main model for n-way classification task
        task_loss_fn : Optional[Callable]. Default to None
            loss function for evaluating the output of a model to the target in data
        kwargs : Dict
            other specifications for model definition and training protocol
            num_tasks_per_iter
            lr_meta
            lr_task
            log_every,
            conv_nf_list : List[int]
              list of number of filters for the convnet; this convnet encodes an image to
              a vector of length `dim_h
              To make the meta-model's output h to be invariant to the order of images in
              in input (set of images), the `forward` method adds up the h from each image
              and use it as the final latent representation of the input (support) set
            fc_nf_list : List[int]
              number of neurons in fully-connected layers, which receives the output of convnet
              (flatten to 1dim vector); its last layer will output `dim_h` sized vector

        """
        super().__init__()
        self.k_shot = k_shot
        self.n_way = n_way
        self.num_tasks_per_iter = kwargs.get('num_tasks_per_iter', 1)

        # data sampler for kshot tasks
        self.sampler = KShotSampler()
        self.lr_task = kwargs.get('lr_task', 0.2)
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()

        self.lr_meta = kwargs.get('lr_meta', 0.3)

        in_h, in_w, in_c = in_shape
        self.device = device
        # log config
        self.log_every = kwargs.get('log_every', 10) #eps

        # convnet as encoder of input
        self.conv_nf_list = kwargs.get('conv_nf_list', [64,64,64,64])
        self.has_bn = kwargs.get('has_bn', True)
        self.act_fn = kwargs.get('act_fn', nn.ReLU())
        # self.convnet = conv_blocks(in_c, self.conv_nf_list, self.has_bn, self.act_fn)

        # shape of the output of the convnet
        self.last_h = int(in_h / (2 ** len(self.conv_nf_list)))
        self.last_w = int(in_w / (2 ** len(self.conv_nf_list)))
        self.dim_flatten = self.last_h * self.last_w * self.conv_nf_list[-1]

        # fc-layer to map feature to output score vector
        # self.fc = nn.Linear(self.dim_flatten, self.n_way)

        self.net = net or nn.Sequential(
            conv_blocks(in_c, self.conv_nf_list, self.has_bn, self.act_fn),
            nn.Flatten(start_dim=1),
            nn.Linear(self.dim_flatten, self.n_way, bias=True)
        )

        # initial set all grad to zero
        # this step is needed to manually add the local gradient to global model
        init_zero_grad(self.parameters())

        # optimizer for meta-parmaeter
        self.optim_meta = optim.Adam(self.parameters(), lr=self.lr_meta)

    def meta_params(self) -> Dict[str, nn.Parameter]:
        """Returns a Dict of current model's parameter name: value (as nn.Parameter).
        Use it to the current states of theta, the meta-parameters
        """
        return {n: p for n, p in self.named_parameters()}

    def forward(self,
                batch_x: Union[torch.Tensor,List[torch.Tensor]],
                ) -> torch.Tensor:
        return self.net(batch_x)

    def forward_set(self,
                set_x: Union[torch.Tensor,List[torch.Tensor]],
                ) -> torch.Tensor:
        """ Encoding function for a set of inputs so that the output is invariant
        to the permutation in the input elements
        todo
        :param set_x:
        :return:
        """
        outs = []
        for x in set_x:
            x.unsqueeze_(dim=0)
            # x = torch.unsqueeze(x, 0)
            outs.append(self.encoder(x))  #todo: add self.encoder
        outs = torch.stack(outs)
        #todo: use self.fc to map feature maps to class scores
        return outs.sum(0)

    def initialize_task_model(self):
        """Returns a dict[name,nn.Parameter] with each parameter cloned from
        curent  model state"""
        #       return {n:p.clone() for p in self.meta_params()}
        return deepcopy(self)

    def run_task_model(self,
                       task_model: nn.Module,
                       batch: Tuple[torch.Tensor, torch.Tensor],
                       mode: str,
                       optim_task: Optional[torch.optim.Optimizer] = None,
                       ) -> Tuple[torch.Tensor, float]:
        """Fit the task_model (ie. learnable tensor) to the support set
        Option1:
        - use functional sgd
          - con: we are choosing the most basic optimization method her
            and also, not really a good abstraction of the general
            optimization-based way to "adapt" the prior (meta-param)
            to this specific task
          - pro: sgd is easy to implement compared to other optimizers like Adam
        Option 2:
        - Use the existing optimizer's in `torch.optim` class
        - Caveat: we must to make sure the updating step of the optimizer
        (aka. optim.step()) operation is properly recorded to the original parameter
        (from which task_model is cloned from)

        Args
        ----
        task_model : nn.Module
          classifier modle to train on the support set
        batch : Tuple[torch.Tensor, torch.Tensor]
            batch of support and query samples
        mode : str ('fit' or 'eval')
        optim_task : Optimizer
          an instantiated torch optimizer; assumed to have the task_model's
          parameters already registered.

        Effects
        -------
        1. The value of the task_model (tensor) is updated as a result of the optimization
        steps.
        2. Each p.grad foro p in task_model.parameters() is filled with the gradients
        from the loss on this support.
        That is, we have updated the task_model's state based on the loss, but have not
        zero'ed out the gradients used for the update in case we may use this gradients
        to update the original theta parameter

        Returns
        -------
        loss, acc : torch.Tensor, float
        """
        assert mode.lower() in ['fit', 'eval'], f'mode must be either fit or eval: {mode}'

        # initialize an inner-optimizer if 'fit' mode
        if mode == 'fit':
            optim_task = optim_task or optim.SGD(task_model.parameters(), lr=self.lr_task)

        imgs, targets = batch
        preds = task_model(imgs)  # scores
        loss = self.task_loss_fn(preds, targets)
        acc = (preds.argmax(dim=1) == targets).sum() / len(preds)
        # --- debug
        # print('=== Task model: inside run_task_model ===')
        # for n, p in task_model.named_parameters():
        #     print(n, p.grad_fn)
        # breakpoint()

        # update the state of task_model if inner-training
        if mode == 'fit':
            optim_task.zero_grad()
            loss.backward()
            optim_task.step()

        # note: now the task_model has updated values in its parameter tensors
        # also, each p.grad is filled with the gradients from the loss on this support
        return loss, acc

    def outer_step(self,
                   meta_dset: KShotImageDataset,
                   step_idx: int,
                   mode='train',
                   verbose: bool=False,
                   show: bool=False,
                   idx2str: Optional[Dict[int,str]]=None,
                   ):
        """ Single update step for the meta-parameter theta
        Args
        ----
        id2str : Optional[Dict[int,str]]
            class label index to string dictionary; used when logging
            sample of support/query to tensorboard logger

        """
        # Subsample tasks (total M number of tasks)
        # print(f"\n === outer-step: {step_idx} ===")
        if (step_idx+1) %self.log_every == 0:
            print(step_idx+1, end="...")
        self.optim_meta.zero_grad()
        loss = 0.0
        accs_spt = []
        accs_q = []
        for m in range(self.num_tasks_per_iter):
            # Prepare support and query sets
            task_set, global2local = meta_dset.sample_task_set(self.n_way)
            local2global = {v: k for k, v in global2local.items()}
            sample = self.sampler.get_support_and_query(task_set,
                                                   num_per_class=self.k_shot,
                                                   shuffle=True,
                                                   collate_fn=torch.stack,
                                                   global_id2local_id=global2local)
            support, query = sample['support'], sample['query'] #support/query: Tuple(batch_x, batch_y)
            batch_x_spt, batch_y_spt = support
            batch_x_q, batch_y_q = query

            # Show support, query
            if verbose and (step_idx+1)%self.log_every == 0:
                print('batch_x_spt, batch_y_spt: ', batch_x_spt.shape, batch_y_spt.shape)
                print('Num of imgs per class in support: ', Counter(support[1].numpy()))
                print('Num of imgs per class in query: ', Counter(query[1].numpy()))
            if show and (step_idx+1)%self.log_every == 0:
                # todo: log to tensorboard
                titles = None # todo
                # titles = [idx2str[local2global[y.item()]] for y in batch_y_spt]
                show_timgs(batch_x_spt,
                           title='support',
                           titles=titles)
                # titles = [idx2str[local2global[y.item()]] for y in batch_y_q]
                show_timgs(batch_x_q,
                           title='query',
                           titles=titles)

            # Turn the list into a tensor, aka. a batch of images, targets for L(theta|support)
            # initialize a task-learner
            task_model = self.initialize_task_model()
            # -- debug
            # print('=== Task model: before run_task_model ===')
            # for n, p in task_model.named_parameters():
            #     print(n, p.flatten(0)[:3], p.grad_fn, p.grad is None)

            # fit the task-learner to support  (inner-loop)
            # -- effect: in-place updates of task_learner's parameters
            _, acc_spt = self.run_task_model(task_model, support, mode='fit')
            accs_spt.append(acc_spt)
            # --- debug
            # print('=== Task model: after run_task_model ===')
            # for n, p in task_model.named_parameters():
            #     print(n, p.flatten(0)[:3], p.grad_fn,  p.grad.flatten(0)[:3])
            # breakpoint()

            # add gradients from inner-loop manully
            # for name, p in self.named_parameters():
            #     p.grad += getattr(task_model, name).grad
            for p_global, p_local in zip(self.parameters(), task_model.parameters()):
                p_global.grad += p_local.grad

            # zero out local modle's grad
            zero_grad(task_model.parameters()) #todo check

            # compute the loss of the fitted task-learner on query
            loss_q, acc_q = self.run_task_model(task_model, query, mode='eval') #todo: why is gradient not being accum?
            accs_q.append(acc_q)
            loss += loss_q

        loss /= self.num_tasks_per_iter
        # # -- debug
        # print('--- meta-model: before update')
        grads_prev = get_grad_ckpt(self)
        # for n, p in self.named_parameters():
        #     print(n, p.flatten(0)[:3], end=', ')
        #     if p.grad is not None:
        #         print(p.grad.flatten()[:3])
        #     else:
        #         print('p.grad is None')
        #     break

        # set task_learner's grad to zero

        # update meta-learner
        # self.optim_meta.zero_grad() #should not do this here! it will zero-out what we accmulated from inner-loops
        # print('loss: ', loss),
        # breakpoint()
        loss.backward() #todo: why is gradient not being accum to self.paramters.grad's?
        # # -- debug
        # print('--- meta-model: after backprop')
        grads_after_bp = get_grad_ckpt(self)
        # for n, p in self.named_parameters():
        #     print(n, p.flatten(0)[:3], end=', ')
        #     if p.grad is not None:
        #         print(p.grad.flatten()[:3])
        #     else:
        #         print('p.grad is None')
        #     break
        self.optim_meta.step()
        # # -- debug
        # print('--- meta-model: after optim step')
        grads_after_update = get_grad_ckpt(self)
        # for n, p in self.named_parameters():
        #     print(n, p.flatten(0)[:3], end=', ')
        #     if p.grad is not None:
        #         print(p.grad.flatten()[:3])
        #     else:
        #         print('p.grad is None')
        #     break
        # breakpoint()


        # -- debug: check if gradients are changing as expected
        # Expected:
        # 1. after backprop on loss, model's grad should change
        # 2. does optim.step() zeros out the gradients of its registered parameter tensors?
        #   -- if so, ater optim.step(), model's grad should be zeroed out
        # print("-- check: grad before and after backprop")
        # print(has_same_values(grads_prev, grads_after_bp))
        #
        # print("-- check: grad after-backward and after-optim-step")
        # print(has_same_values(grads_after_bp, grads_after_update))

        # do logging
        # print("i_meta: ", step_idx)
        acc_spt = np.mean(accs_spt)
        acc_q = np.mean(accs_q)
        # print(f"--- loss/{mode}", loss.item())
        # print(f"--- acc_spt/{mode}", acc_spt)
        # print(f"--- acc_q/{mode}", acc_q)

#     # do logging
#     self.log(f"loss/{mode}", loss.item())
#     self.log(f"acc_spt/{mode}", accs_spt.mean())
#     self.log(f"acc_q/{mode}", accs_q.mean())
        return {'loss': loss, 'acc_spt': acc_spt, 'acc_q': acc_q}
#   # todo: training_step(self, batch, batch_idx)
#   def training_step(self, meta_dset: KShotImageDataset):
#     self.outer_step(meta_dset, mode='train')
#     return None

#   def validation_step(self, meta_dset: KShotImageDataset):
#     self.outer_step(meta_dset, mode='train')
#     return None


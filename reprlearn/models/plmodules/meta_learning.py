from .types_ import *
from copy import deepcopy
from collections import Counter
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from reprlearn.models.convnet import conv_blocks
from reprlearn.models.utils import init_zero_grad, get_grad_ckpt, get_model_ckpt, \
    has_same_values, zero_grad, model_requires_grad
from reprlearn.data.datasets.kshot_dataset import KShotImageDataset
from reprlearn.data.samplers.kshot_sampler import KShotSampler
from reprlearn.visualize.utils import show_timgs
from IPython.core.debugger import set_trace as breakpoint


class MAML(pl.LightningModule):

    def __init__(self,
                 in_shape: List[int],
                 k_shot: int,
                 n_way: int,
                 net: Optional[nn.Module]= None,
                 task_loss_fn: Optional[Callable] = None,
                 use_averaged_meta_loss: bool=True,
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
            lr_meta (default 0.4)
            lr_task (default 1e-3)
            num_inner_steps (default 1)
            log_every (default 10)
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

        # data sampler for kshot tasks
        self.sampler = KShotSampler()
        self.lr_task = kwargs.get('lr_task', 1e-3)
        self.num_inner_steps = kwargs.get('num_inner_steps', 1)
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()

        self.lr_meta = kwargs.get('lr_meta', 0.4)
        self.use_averaged_meta_loss = use_averaged_meta_loss

        in_h, in_w, in_c = in_shape
        # self.device = device
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
        # we handle optimization ourselves, w/o relying on pl abstraction
        self.automatic_optimization = False # Important: This property activates manual optimization
        # self.optim_meta = optim.Adam(self.parameters(), lr=self.lr_meta)

    @property
    def name(self):
        return f'MAML-{self.k_shot}shot-{self.n_way}way-lr_meta:{self.lr_meta:.3f}-lr_task:{self.lr_task:.3f}'

    @staticmethod
    def add_model_specific_args(parent_parser):
        #todo
        return parent_parser

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr_meta)

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
                       batch_data: Tuple[torch.Tensor, torch.Tensor],
                       create_graph: bool,
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
        batch_data : Tuple[torch.Tensor, torch.Tensor]
            batch of support and query samples
        create_graph : bool
            if True, we assume it's required to record the forward computation of
              the task_model on the `batch_data`, and verifies this by the output
              of task_model(inputs) has `requires_grad` of True.
            if False, the last loss does not need be learnable, so we turn off
            the last assertion that checks if the loss is learnable.

        Returns
        -------
        loss, acc : torch.Tensor, float
        """
        imgs, targets = batch_data
        preds = task_model(imgs)  # scores

        # >>> debug >>>
        # print("Task_model requires grad: ", model_requires_grad(task_model))
        # assert preds.requires_grad, "preds from the run_task must be trainable tensor"
        # print('create_graph: ', create_graph)
        # print('preds requires_grad: ', preds.requires_grad)
        assert create_graph == preds.requires_grad, \
            f"if create_graph is {create_graph}, preds also should be {create_graph}"
        # <<< debug <<<

        loss = self.task_loss_fn(preds, targets)
        acc = (preds.argmax(axis=-1) == targets).sum().cpu() / len(preds)

        return loss, acc

    def fit_to_task(self,
                    batch_data: Tuple[torch.Tensor, torch.Tensor],
                    optim_fn: Optional[Callable] = None,
                    ) -> Tuple[nn.Module, torch.Tensor, float]:
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
        batch_data : Tuple[torch.Tensor, torch.Tensor]
            batch of support and query samples
        mode : str ('fit' or 'eval')
        optim_task : Optimizer
          an instantiated torch optimizer; assumed to have the task_model's
          parameters already registered.

        Effects
        -------
        1. The value of the task_model (tensor) is updated as a result of the optimization
        steps.
        2. Each p.grad forop in task_model.parameters() is filled with the gradients
        from the loss on this support.

        That is, we have updated the task_model's state based on the loss, but have not
        zero'ed out the gradients used for the update in case we may use this gradients
        to update the original theta parameter.

        Summary of the effects
        ----------
        After this method returns,
        1. task_model has parameters fitted (via optimization
        steps of `self.num_inner_steps` on the same batch of datapts,
        2. every parameter of task_model's grad has gradients of the loss on
        the batch of datapts, averaged across the  number of optimization updates
        Note that we haven't taken the optimization step (using the
        accumulated averaged gradients from this task data (either support or query)),
        which will later need to be done to update the meta-parameter.

        Note
        -----
        in meta-learning, we need to always (ie. both during training and val/testing)
        be able to compute the gradient during the fast-adapt (aka. fit-to-task,
        aka, inner-optimization) phase.
        However, in pytorch-lighting's Validation Loop (which calls the `validation_step`)
        `torch.set_grad_enabled` is turned off while running the `validation_step`
        which prevents the fast-adaption which is required in the validation step
        (fast-adaptation is required for the training step, (on support set), but
        the training_step is done under the context of `torch.set_grad_enabled(True)`,
        so we have no problem in automatically computing the gradient)
        So, it's important to make sure the fast-adaptation's steps' computation
        continues being tracked during the validation step, which can be ensured
        by explicitly setting the context manager of `with torch.set_grad_enable(True)`

        todo:
        currently this method will not do a proper inner-loop updates when
        `num_ineer_steps`>1 because we are not zeroing out the gradient
        from the previous optim step before updating for this step (so that
        the optimizer takes a step based on accumulated gradient (uptil this inner step)
        rather than just this inner update iteration)

        so, if we want to make it work with num_inner_steps >1,
        we should update the global/meta-parameters inside this function,
        after `loss_i.backward()` by manually adding the gradients from this inner
        step to the meta-parameter's grad fields. And additionally, we should comment out
        `optim_task.zero_grad()` inside the inner loop.

        Returns
        -------
        fit_task_model, loss_spt, acc_spt : nn.Module, torch.Tensor, float
        """
        assert self.num_inner_steps == 1, 'only num_inner_steps of 1 is currently support'

        # initialize an inner-optimizer if 'fit' mode
        task_model = deepcopy(self.net) #deepcopy(self)
        optim_task = optim_fn or optim.SGD(task_model.parameters(), lr=self.lr_task)
        optim_task.zero_grad() #todo: do it here when num inner >1

        with torch.set_grad_enabled(True):  # see "Note"
            # inner-loop iteration
            loss_spt = 0.0
            acc_spt = 0.0
            for i in range(self.num_inner_steps):
                loss_i, acc_i = self.run_task_model(task_model, batch_data, create_graph=True)
                loss_spt += loss_i
                acc_spt += acc_i

                # optim_task.zero_grad() # nb: commented out bc we need to accumulate
                # the gradients so that we can use the average gradients (avg over inner-steps)
                # when flowing back to meta-param
                loss_i.backward()
                optim_task.step()
            # update the state of task_model if inner-training
            loss_spt /= self.num_inner_steps
            acc_spt /= self.num_inner_steps

            # note: now the task_model has updated values in its parameter tensors
            # also, each p.grad is filled with the gradients from the loss on this support (accumulated during iters)
            return task_model, loss_spt, acc_spt

    def training_step(self, batch, batch_idx) -> Dict[str,Any]:
        """Given a batch of task-sets, output a loss tensor or a dict for the
        meta-update step based on the averaged loss on each query in the batch.
        We compute the average loss_q from each task_set in this batch and take
        gradient of that average batch loss_q to compute the graident for updating
        the meta-parameter at this step.
        Current implementation  is a non-vectorized mini_batch_loss computation,
        ie. we loop over the batch of task-sets and compute the mean of each loss_q
        of each query.

        Args
        ----
        batch : List[Dict[str,Tuple[torch.Tensor, torch.Tensor]]]
         a list of task_sets, which is a dict containing support and query sample sets

        Returns
        dict_loss : Dict[str, Union[torch.Tensor, float]
            a dictionary containing support loss, query loss, and support-acc,
            query-acc based on the averaged loss/acc values (averaged over the
            task-sets in the batch of task-sets)
        -------

        """
        print(batch_idx, end='...') #meta-iter
        self.optimizers().zero_grad() #important!

        losses_spt, accs_spt = [], []
        losses_q, accs_q = [], []
        for task_set in batch:
            batch_spt, batch_q = task_set['support'], task_set['query']
            batch_spt_x, batch_spt_y = batch_spt
            batch_q_x, batch_q_y = batch_q

            # now, use the support/query as if it's the regular batch of datapts
            fit_task_model, loss_spt, acc_spt = self.fit_to_task(batch_spt)
            losses_spt.append(loss_spt.item())
            accs_spt.append(acc_spt)

            # debug
            # for n,p in fit_task_model.named_parameters(): print(n,p.grad.flatten()[:2])
            # for n,p in self.named_parameters(): print(n,p.grad.flatten()[:2])
            # ---
            # add gradients from inner-loop manually
            for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
              p_global.grad += p_local.grad / self.num_inner_steps  # todo: p_local.grad / self.num_inner_steps

            # zero out local model's grad
            zero_grad(fit_task_model.parameters())  # todo unnecessary?
            # ---

            # 2. Accumulate this query's loss
            # compute the loss of the fitted task-learner on query
            loss_q, acc_q = self.run_task_model(fit_task_model, batch_q, create_graph=True)  # todo: check the gradient is being accum
            losses_q.append(loss_q.item())
            accs_q.append(acc_q)

            # loss += loss_q
            # Update: 2022-01-08
            # - instead, manually add the gradient loss_q wrt mcurrent meta-paameter to mata-param's grad
            # directly, at each task-step (we will take the optimization step after all batch of tasks, though -- so no difference in doing loss =+= loss_q and them backpro + optim step)
            # loss_q.backward()
            # loss is sum of loss_q_i's; note: not an average (e.g. loss /=self.num_tasks_per_iter)
            if self.use_averaged_meta_loss:
                loss_q /= len(batch) #aka. self.num_tasks_per_iter
            self.manual_backward(loss_q) # because we are handling the graident to the model param manually in pl
            # ^ we backprop loss_q gradients for individual task-set and accumulate the grads over the batch of task-sets

            grads_prev = get_grad_ckpt(self)
            # ---
            # add gradients from loss_q to meta-param.grad's manually
            for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
                p_global.grad += p_local.grad
            grads_after = get_grad_ckpt(self)
            assert not has_same_values(grads_prev, grads_after), \
                "Meta-param's grad's should change after adding gradients from loss_q"

            # todo: we can safely delete task and fit-task models since all of their gradient contributions are manually
            # added to meta-parameter
            # del task_model, fit_task_model

        # update meta-learner
        # loss is **sum** of loss_q_i's if not self.use_averaged_meta_loss
        # If self.use_averaged_meat_oss: loss /=self.num_tasks_per_iter)

        # self.optim_meta.zero_grad() #should not do this here! it will zero-out
        # what we have accumulated from inner-loop optimization steps and loss_q computation
        # self.optim_meta.step()
        self.optimizers().step() #manual optimization in pl

        # logging
        # print("meta-step: ", batch_idx)
        loss_spt = np.mean(losses_spt)
        loss_q = np.mean(losses_q)
        acc_spt = np.mean(accs_spt)
        acc_q = np.mean(accs_q)

        self.log(f"train/loss_spt",loss_spt)
        self.log(f"train/loss_q", loss_q)
        self.log(f"train/acc_spt", acc_spt)
        self.log(f"train/acc_q", acc_q)
        return {'loss_spt': loss_spt, 'loss_q': loss_q,
                'acc_spt': acc_spt, 'acc_q': acc_q}

    # def on_train_batch_start(
    #     self, batch: Any, batch_idx: int, dataloader_idx: int
    # ) -> None:
    #     print("Train batch starts >>>")
    #
    # def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    #     print("<<< Train batch ends ")
    #
    # def on_validation_epoch_start(self) -> None:
    #     self.ckpt_val_start =  get_model_ckpt(self)
    #     print("Validation ep starts >>>")
    #
    # def on_validation_epoch_end(self) -> None:
    #     self.ckpt_val_end =  get_model_ckpt(self)
    #     assert has_same_values(self.ckpt_val_start,self.ckpt_val_end)
    #     del self.ckpt_val_start, self.ckpt_val_end
    #     print("<<< Validation ep ends")

    def validation_step(self, batch, batch_idx) -> Dict[str,Any]:
        """
        Note
        -----
        in meta-learning, we need to always (ie. both during training and val/testing)
        be able to compute the gradient during the fast-adapt (aka. fit-to-task,
        aka, inner-optimization) phase.
        However, in pytorch-lighting's Validation Loop (which calls the `validation_step`)
        `torch.set_grad_enabled` is turned off while running the `validation_step`
        which prevents the fast-adaption which is required in the validation step
        (fast-adaptation is required for the training step, (on support set), but
        the training_step is done under the context of `torch.set_grad_enabled(True)`,
        so we have no problem in automatically computing the gradient)
        So, it's important to make sure the fast-adaptation's steps' computation
        continues being tracked during the validation step, which can be ensured
        by explicitly setting the context manager of `with torch.set_grad_enable(True)`

        todo:
        - [ ] check that the gradient on the model parameter don't change at all
            during this whole step

        """
        losses_spt, accs_spt = [], []
        losses_q, accs_q = [], []
        for task_set in batch:
            batch_spt, batch_q = task_set['support'], task_set['query']
            batch_spt_x, batch_spt_y = batch_spt
            batch_q_x, batch_q_y = batch_q

            # now, use the support/query as if it's the regular batch of datapts
            fit_task_model, loss_spt, acc_spt = self.fit_to_task(batch_spt)
            losses_spt.append(loss_spt.item())
            accs_spt.append(acc_spt)

            # ---
            # add gradients from inner-loop manully
            # for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
            #     p_global.grad += p_local.grad / self.num_inner_steps  # todo: p_loca.grad / self.num_inner_steps

            # zero out local model's grad
            # zero_grad(fit_task_model.parameters())  # todo unnecessary?
            # ---

            # 2. Accumulate this query's loss
            # compute the loss of the fitted task-learner on query
            loss_q, acc_q = self.run_task_model(fit_task_model, batch_q, create_graph=False)  # todo: why is gradient not being accum?
            losses_q.append(loss_q.item())
            accs_q.append(acc_q)

            # loss += loss_q
            # Update: 2022-01-08
            # - instead, manually add the gradient loss_q wrt mcurrent meta-paameter to mata-param's grad
            # directly, at each task-step (we will take the optimization step after all batch of tasks, though -- so no difference in doing loss =+= loss_q and them backpro + optim step)
            # loss_q.backward()

            # grads_prev = get_grad_ckpt(self)
            # --- no accumulating gradients to meta-parameter needed as we are not learning
            # add gradients from loss_q to meta-param.grad's manually
            # for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
            #     p_global.grad += p_local.grad
            # grads_after = get_grad_ckpt(self)
            # assert not has_same_values(grads_prev,
            #                            grads_after), "Meta-param's grad's should change after adding gradients from loss_q"

            # todo: we can safely delete task and fit-task models since all of their gradient contributions are manually
            # added to meta-parameter
            # del task_model, fit_task_model

        # no update of meta-learner

        # logging
        # print("meta-step: ", batch_idx)
        loss_spt = np.mean(losses_spt)
        loss_q = np.mean(losses_q)
        acc_spt = np.mean(accs_spt)
        acc_q = np.mean(accs_q)

        self.log(f"val/loss_spt", loss_spt)
        self.log(f"val/loss_q", loss_q)
        self.log(f"val/acc_spt", acc_spt)
        self.log(f"val/acc_q", acc_q)
        return {'loss_spt': loss_spt, 'loss_q': loss_q,
                'acc_spt': acc_spt, 'acc_q': acc_q}

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """
        Note
        -----
        in meta-learning, we need to always (ie. both during training and val/testing)
        be able to compute the gradient during the fast-adapt (aka. fit-to-task,
        aka, inner-optimization) phase.
        However, in pytorch-lighting's Validation Loop (which calls the `validation_step`)
        `torch.set_grad_enabled` is turned off while running the `validation_step`
        which prevents the fast-adaption which is required in the validation step
        (fast-adaptation is required for the training step, (on support set), but
        the training_step is done under the context of `torch.set_grad_enabled(True)`,
        so we have no problem in automatically computing the gradient)
        So, it's important to make sure the fast-adaptation's steps' computation
        continues being tracked during the validation step, which can be ensured
        by explicitly setting the context manager of `with torch.set_grad_enable(True)`

        todo:
        - [ ] check that the gradient on the model parameter don't change at all
            during this whole step

        """
        losses_spt, accs_spt = [], []
        losses_q, accs_q = [], []
        for task_set in batch:
            batch_spt, batch_q = task_set['support'], task_set['query']
            batch_spt_x, batch_spt_y = batch_spt
            batch_q_x, batch_q_y = batch_q

            # now, use the support/query as if it's the regular batch of datapts
            fit_task_model, loss_spt, acc_spt = self.fit_to_task(batch_spt)
            losses_spt.append(loss_spt.item())
            accs_spt.append(acc_spt)

            # ---
            # add gradients from inner-loop manully
            # for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
            #     p_global.grad += p_local.grad / self.num_inner_steps  # todo: p_loca.grad / self.num_inner_steps

            # zero out local model's grad
            # zero_grad(fit_task_model.parameters())  # todo unnecessary?
            # ---

            # 2. Accumulate this query's loss
            # compute the loss of the fitted task-learner on query
            loss_q, acc_q = self.run_task_model(fit_task_model, batch_q, create_graph=False)  # todo: why is gradient not being accum?
            losses_q.append(loss_q.item())
            accs_q.append(acc_q)

            # loss += loss_q
            # Update: 2022-01-08
            # - instead, manually add the gradient loss_q wrt mcurrent meta-paameter to mata-param's grad
            # directly, at each task-step (we will take the optimization step after all batch of tasks, though -- so no difference in doing loss =+= loss_q and them backpro + optim step)
            # loss_q.backward()

            # grads_prev = get_grad_ckpt(self)
            # --- no accumulating gradients to meta-parameter needed as we are not learning
            # add gradients from loss_q to meta-param.grad's manually
            # for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
            #     p_global.grad += p_local.grad
            # grads_after = get_grad_ckpt(self)
            # assert not has_same_values(grads_prev,
            #                            grads_after), "Meta-param's grad's should change after adding gradients from loss_q"

            # todo: we can safely delete task and fit-task models since all of their gradient contributions are manually
            # added to meta-parameter
            # del task_model, fit_task_model

        # no update of meta-learner

        # logging
        # print("meta-step: ", batch_idx)
        loss_spt = np.mean(losses_spt)
        loss_q = np.mean(losses_q)
        acc_spt = np.mean(accs_spt)
        acc_q = np.mean(accs_q)

        self.log(f"val/loss_spt", loss_spt)
        self.log(f"val/loss_q", loss_q)
        self.log(f"val/acc_spt", acc_spt)
        self.log(f"val/acc_q", acc_q)
        return {'loss_spt': loss_spt, 'loss_q': loss_q,
                'acc_spt': acc_spt, 'acc_q': acc_q}


# Archive
#     def outer_step(self,
#                    meta_dset: KShotImageDataset,
#                    step_idx: int,
#                    mode='train',
#                    verbose: bool=False,
#                    show: bool=False,
#                    idx2str: Optional[Dict[int,str]]=None,
#                    ):
#         """ Single update step for the meta-parameter theta
#         Args
#         ----
#         meta_dset : KShotImageDataset
#             meta-train dataset to sample task-sets from
#         step_idx : int
#             index for outer update step
#
#         id2str : Optional[Dict[int,str]]
#             class label index to string dictionary; used when logging
#             sample of support/query to tensorboard logger
#
#         """
#         # Subsample tasks (total M number of tasks)
#         # print(f"\n === outer-step: {step_idx} ===")
#         if (step_idx+1) % self.log_every == 0:
#             print(step_idx+1, end="...")
#         self.optim_meta.zero_grad()
#         loss = 0.0
#         losses_spt= []
#         losses_q = []
#         accs_spt = []
#         accs_q = []
#         for m in range(self.num_tasks_per_iter):
#             # Prepare support and query sets
#             task_set, global2local = meta_dset.sample_task_set(self.n_way)
#             local2global = {v: k for k, v in global2local.items()}
#             sample = self.sampler.get_support_and_query(task_set,
#                                                    num_per_class=self.k_shot,
#                                                    shuffle=True,
#                                                    collate_fn=torch.stack,
#                                                    global_id2local_id=global2local)
#             support, query = sample['support'], sample['query'] #support/query: Tuple(batch_x, batch_y)
#             batch_x_spt, batch_y_spt = support
#             batch_x_q, batch_y_q = query
#
#             # Show support, query
#             if verbose and (step_idx+1)%self.log_every == 0:
#                 print('batch_x_spt, batch_y_spt: ', batch_x_spt.shape, batch_y_spt.shape)
#                 print('Num of imgs per class in support: ', Counter(support[1].numpy()))
#                 print('Num of imgs per class in query: ', Counter(query[1].numpy()))
#             if show and (step_idx+1)%self.log_every == 0:
#                 # todo: log to tensorboard
#                 titles = None # todo
#                 # titles = [idx2str[local2global[y.item()]] for y in batch_y_spt]
#                 show_timgs(batch_x_spt,
#                            title='support',
#                            titles=titles)
#                 # titles = [idx2str[local2global[y.item()]] for y in batch_y_q]
#                 show_timgs(batch_x_q,
#                            title='query',
#                            titles=titles)
#
#             # Turn the list into a tensor, aka. a batch of images, targets for L(theta|support)
#             # initialize a task-learner
#             # fit the task-learner to support  (inner-loop)
#             # -- effect: in-place updates of task_learner's parameters
#             fit_task_model, loss_spt, acc_spt = self.fit_to_task(support)
#             losses_spt.append(loss_spt.item())
#             accs_spt.append(acc_spt)
#
#             # ---
#             # add gradients from inner-loop manully
#             for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
#                 p_global.grad += p_local.grad / self.num_inner_steps  #todo: p_loca.grad / self.num_inner_steps
#
#             # zero out local model's grad
#             zero_grad(fit_task_model.parameters())  # todo unnecessary?
#             # ---
#
#             # compute the loss of the fitted task-learner on query
#             loss_q, acc_q = self.run_task_model(fit_task_model, query)  # todo: why is gradient not being accum?
#             losses_q.append(loss_q.item())
#             accs_q.append(acc_q)
#
#             # loss += loss_q
#             # Update: 2022-01-08
#             # - instead, manually add the gradient loss_q wrt mcurrent meta-paameter to mata-param's grad
#             # directly, at each task-step (we will take the optimization step after all batch of tasks, though -- so no difference in doing loss =+= loss_q and them backpro + optim step)
#             loss_q.backward()
#
#             grads_prev = get_grad_ckpt(self)
#             # ---
#             # add gradients from loss_q to meta-param.grad's manually
#             for p_global, p_local in zip(self.parameters(), fit_task_model.parameters()):
#                 p_global.grad += p_local.grad
#             grads_after = get_grad_ckpt(self)
#             assert not has_same_values(grads_prev, grads_after), "Meta-param's grad's should change after adding gradients from loss_q"
#
#
#             # todo: we can safely delete task and fit-task models since all of their gradient contributions are manually
#             # added to meta-parameter
#             # del task_model, fit_task_model
#
#
#         # update meta-learner
#         # loss /= self.num_tasks_per_iter
#         # self.optim_meta.zero_grad() #should not do this here! it will zero-out what we accmulated from inner-loops
#         # loss.backward() #todo: why is gradient not being accum to self.paramters.grad's?
#         # update meta-parameters using the manually accumulated gradients
#         # with torch.no_grad():
#         #     for p in self.parameters():
#         #         p /= self.num_tasks_per_iter
#         self.optim_meta.step()
#
#         # do logging
#         # print("i_meta: ", step_idx)
#         loss_spt = np.mean(losses_spt)
#         loss_q = np.mean(losses_q)
#         acc_spt = np.mean(accs_spt)
#         acc_q = np.mean(accs_q)
#         # print(f"--- loss/{mode}", loss.item())
#         # print(f"--- acc_spt/{mode}", acc_spt)
#         # print(f"--- acc_q/{mode}", acc_q)
#
# #     # do logging
# #     self.log(f"loss/{mode}", loss.item())
# #     self.log(f"acc_spt/{mode}", accs_spt.mean())
# #     self.log(f"acc_q/{mode}", accs_q.mean())
#         return {'loss_spt': loss_spt, 'loss_q': loss_q,
#                 'acc_spt': acc_spt, 'acc_q': acc_q}


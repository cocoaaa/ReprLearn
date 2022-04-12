import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, List, Optional

class DummyConvNet(nn.Module):
    # Convolutional neural network (two convolutional layers)

    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class DummyConvNet2(nn.Module):
    # Convolutional neural network (two convolutional layers)

    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out



class FourLayerConvNet(nn.Module):
    # convnet used in Vinloly 2016 and MAML 2018
    # References:
    # original code by cbfinn:
    #   https://tinyurl.com/y53r9uaz
    # pytorch version by dragen1860:
    #   https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py#L16
    conv_config = {'kernel_size': 3, 'stride': 2, 'padding': 1}

    def __init__(self,
                 in_shape: List[int],
                 num_classes: int,
                 nf_list: Optional[List[int]] = None,
                 act_fn: Optional[Callable] = nn.ReLU()
                 ):
        """

        :param in_shape:  h,w,num_channels
        :param num_classes: number 0f classes in target varible
        :param nf_list: list of number of filters at each conv-layer
        :param act_fn:
        """
        super().__init__()
        self.nf_list = nf_list or [64, 64, 64, 64]
        self.num_classes = num_classes
        self.in_h, self.in_w, self.in_c = in_shape
        self.out_h = self.in_h // (2**4)
        self.out_w = self.in_w // (2**4)
        self.dim_flatten = self.nf_list[-1] * self.out_h * self.out_w
        print(self.out_h, self.out_h, self.nf_list[-1],self.dim_flatten)

        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.nf_list[0], **self.conv_config),
            act_fn,
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(self.nf_list[0]),
            nn.Conv2d(self.nf_list[0], self.nf_list[1], **self.conv_config),
            act_fn
        )

        self.block3 = nn.Sequential(
            nn.BatchNorm2d(self.nf_list[1]),
            nn.Conv2d(self.nf_list[1], self.nf_list[2], **self.conv_config),
            act_fn
        )
        self.block4 = nn.Sequential(
            nn.BatchNorm2d(self.nf_list[2]),
            nn.Conv2d(self.nf_list[2], self.nf_list[3], **self.conv_config),
            act_fn
        )
        self.norm4fc = nn.BatchNorm2d(self.nf_list[-1])

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(self.dim_flatten), # todo: not sure if it's okay to add it here
            nn.Linear(self.dim_flatten, self.num_classes, bias=True)
        )

    def forward(self, x: torch.Tensor):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.norm4fc(out)
        out = out.view(len(out), -1)
        return self.fc(out)



def conv_block(
        in_channels: int,
        out_channels: int,
        has_bn: bool = True,
        act_fn: Callable = None,
        **kwargs) -> nn.Sequential:
    """
    Returns a conv block of Conv2d -> (BN2d) -> act_fn

    kwargs: (will be passed to nn.Conv2d)
    - kernel_size:int
    - stride: int
    - padding
    - dilation
    - groups
    - bias
    - padding_mode
    """
    # Default conv_kwargs is overwritten by input kwargs
    bias = False if has_bn else True
    conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': bias}
    conv_kwargs.update(kwargs)

    if act_fn is None:
        act_fn = nn.LeakyReLU()

    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, **conv_kwargs)),
        ('bn', nn.BatchNorm2d(out_channels) if has_bn else nn.Identity()),
        ('act', act_fn)
    ]))



def conv_blocks(
        in_channels: int,
        nf_list: List[int],
        has_bn=True,
        act_fn: Optional[Callable]=None,
        **kwargs) -> nn.Sequential:
    """
    Returns a nn.Sequential of conv_blocks, each of which is itself a nn.Sequential
    of Conv2d, (BN2d) and activation function (eg. ReLU(), LeakyReLU())
    """
    if act_fn is None:
        act_fn = nn.LeakyReLU()

    blocks = []
    nfs = [in_channels, *nf_list]
    for i, (in_c, out_c) in enumerate(zip(nfs, nfs[1:])):
        name = f'cb{i}'
        blocks.append(
            (name, conv_block(in_c, out_c, has_bn=has_bn, act_fn=act_fn, **kwargs))
        )

    return nn.Sequential(OrderedDict(blocks))


# conv_net = conv_blocks



def deconv_block(
        in_channels: int,
        out_channels: int,
        has_bn: bool = True,
        act_fn: Callable = None,
        **kwargs
) -> nn.Sequential:
    """
    Returns a deconv block of ConvTranspose2d -> (BN2d) -> act_fn

    kwargs: (will be passed to nn.Conv2d)
    - kernel_size:int
    - stride: int
    - padding
    - output_padding
    """
    # Default conv_kwargs is overwritten by input kwargs
    bias = False if has_bn else True
    deconv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1,
                     'output_padding':1, 'bias': bias}
    deconv_kwargs.update(kwargs)

    if act_fn is None:
        act_fn = nn.LeakyReLU()

    return nn.Sequential(OrderedDict([
        ('deconv', nn.ConvTranspose2d(in_channels, out_channels, **deconv_kwargs)),
        ('bn', nn.BatchNorm2d(out_channels) if has_bn else nn.Identity()),
        ('act', act_fn)
    ]))

def deconv_blocks(
        in_channels: int,
        nf_list: List[int],
        has_bn=True,
        act_fn: Optional[Callable]=None,
        **kwargs) -> nn.Sequential:
    """
    Returns a nn.Sequential of deconv_blocks, each of which is itself a nn.Sequential
    of ConvTransposed, (BN2d) and activation function (eg. ReLU(), LeakyReLU())
    """
    if act_fn is None:
        act_fn = nn.LeakyReLU()

    blocks = []
    # nf_list.insert(0, in_channels)  # don't do this; in-place->changes nf_list outside of this function
    # Instead, make a local variable
    nfs = [in_channels, *nf_list]
    for i, (in_c, out_c) in enumerate(zip(nfs, nfs[1:])):
        name = f'de_cb{i}'
        blocks.append(
            (name, deconv_block(in_c, out_c, has_bn=has_bn, act_fn=act_fn, **kwargs))
        )

    return nn.Sequential(OrderedDict(blocks))



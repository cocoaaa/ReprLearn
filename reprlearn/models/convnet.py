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



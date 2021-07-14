import torch.nn as nn
from typing import List

class Residual(nn.Module):
    """A module that implements a single flow of residual operation for ResNet.
    Each conv layer uses kernel of size 3x3, stride=streids, and padding=1.
    First the input's (h,w) are shrinked by `stride`, then the num of channels
    is increased to out_c via subsequent conv operations.


    input ---> (bn-relu-conv2d) -> z1 ---> (bn-relu-conv2d) -> z2 ---> out
            |                                                   ^
            |                                                   |
            |                                                   +
            ----------------->  (1x1 conv2d) --------------------


    Parameters
    ----------
    stride : int
        Stride parameter of the first conv layer. Use stride = 2 as a way to
        halve the width, height of the input; similar to applying a pooling
        operation.


    use_1x1conv : bool
        Applies the 1x1 conv to the input to match the input's n_channel (in_c)
        to be equal to the output's n_chhanel (out_c), as well as (h,w) adjustment
        by `stride`.

        It must be set to True when the input's num channel or (h,w) need to
        - in_c is different from out_c, or
        - `stride` != 1, or
        - same shape of input and output, but just want to add 1x1 conv operation
        to the input before adding it to the activation after the second conv.

    `forward(x)` returns
    -----------
    out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h/stride, in_w/stride)


    """

    def __init__(self, in_c, out_c,
                 *,
                 stride=1,
                 use_1x1conv=False,
                 act_fn=nn.ReLU(inplace=True),
                 kernel_size=3,
                 padding=1):
        super().__init__()
        self.stride = stride
        self.use_1x1conv = use_1x1conv

        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c,
                               kernel_size=kernel_size, padding=padding, stride=self.stride)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c,
                               kernel_size=kernel_size, padding=padding, stride=1)

        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_c, out_c,
                                   kernel_size=1, padding=0, stride=self.stride)
        self.act_fn = act_fn

    def forward(self, x):
        """
        Returns
        -------
        out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h/stride, in_w/stride)

        """
        z = self.conv1(self.act_fn(self.bn1(x)))
        z = self.conv2(self.act_fn(self.bn2(z)))

        if self.use_1x1conv:
            x = self.conv3(x)
        z = z + x
        return self.act_fn(z)


class Residual_V1(nn.Module):
    """A module that implements a single flow of residual operation for ResNet.
    Each conv layer uses kernel of size 3x3, stride=streids, and padding=1.
    First the input's (h,w) are shrinked by `stride`, then the num of channels
    is increased to out_c via subsequent conv operations.


    input ---> conv2d-bn-relu -> z1 ---> conv2d-bn-----> z2 -> relu -> out
            |                                            ^
            |                                            |
            |                                            +
            ----------------->(1x1 conv2d)----------------


    Parameters
    ----------
    stride : int
        Stride parameter of the first conv layer. Use stride = 2 as a way to
        halve the width, height of the input; similar to applying a pooling
        operation.


    use_1x1conv : bool
        Applies the 1x1 conv to the input to match the input's n_channel (in_c)
        to be equal to the output's n_chhanel (out_c), as well as (h,w) adjustment
        by `stride`.

        It must be set to True when the input's num channel or (h,w) need to
        be adjusted in order to be added to the second conv's output (z2), ie:
        - in_c is different from out_c, or
        - `stride` != 1, or
        - same shape of input and output, but just want to add 1x1 conv operation
        to the input before adding it to the activation after the second conv.

    `forward(x)` returns
    -----------
    out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h/stride, in_w/stride)


    """

    def __init__(self, in_c, out_c,
                 *,
                 stride=1,
                 use_1x1conv=False,
                 act_fn=nn.ReLU(inplace=True),
                 kernel_size=3, padding=1):
        super().__init__()
        self.stride = stride
        self.use_1x1conv = use_1x1conv
        self.conv1 = nn.Conv2d(in_c, out_c,
                               kernel_size=kernel_size, padding=padding, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c,
                               kernel_size=kernel_size, padding=padding, stride=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_c, out_c,
                                   kernel_size=1, padding=0, stride=self.stride)
        self.act_fn = act_fn

    def forward(self, x):
        """
        Returns
        -------
        out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h/stride, in_w/stride)

        """
        z = self.act_fn(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))

        if self.use_1x1conv:
            x = self.conv3(x)
        z = z + x
        return self.act_fn(z)



def get_resnet_block(in_c, out_c, *,
                     n_residuals=2,
                     first_block=False,
                     ) -> List[nn.Module]:
    """
    Each of the subsequent blocks contain 2 residual operations, where the output channel
    is doubled and the resolution (ie. h,w) is halved
    :param in_c:
    :param out_c:
    :param n_residuals:
    :param first_block:
    :return:
    """
    # First residual: In the first block, we don't adjust the (h,w) by half because
    # the input is already processed by a MaxPool layer
    if first_block:
        use_1x1conv = False if in_c == out_c else True
        res0 = Residual(in_c, out_c, stride=1, use_1x1conv=use_1x1conv)

    else:
        res0 = Residual(in_c, out_c, stride=2, use_1x1conv=True)

    block = [res0]
    # Add subsequence residuals
    for i in range(n_residuals - 1):
        block.append(Residual(out_c, out_c,
                              stride=1, use_1x1conv=False))
    return block


class ResNet(nn.Module):
    """
    input -> b1 (conv(1/2)-bn-act) -> resnet_blocks -> out

    Model weights
    -------------
    b1: (conv(1/2)-bn-relu). No extra maxpool(1/2)) unlike original ResNet
        kernel_size: 7x7, padding:3, stride:2 # halves (H,W)
        out_channels = hidden_dims[0]
    resnet_blocks: nn.Sequential of resnet blocks
        Each block has 2 residual units: (bn-relu-conv, bn-relu-conv)

    Returns
    -------
    out : (BS, hidden_dims[-1], last_h, last_w)

    Note
    ----
    Unlike original ResNet which uses MaxPool to halves (H,W) resolutions,
    this network uses convolution with stride=2.
    """
    def __init__(self,
                 in_c: int,
                 hidden_dims: List[int],
                 act_fn=nn.ReLU(inplace=True)):
        super().__init__()

        self.in_c = in_c
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn

        n0 = hidden_dims[0]
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, n0, kernel_size=7, stride=2, padding=3),  # (bs,n0,h/2,w/2)
            nn.BatchNorm2d(n0),
            self.act_fn,
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (bs, n0, h/2, w/2)
        )

        blocks = []
        for i, (in_c, out_c) in enumerate(zip(hidden_dims, hidden_dims[1:])):
            is_first = (i == 0)
            blocks.extend(get_resnet_block(in_c, out_c, first_block=is_first))

        self.resnet_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x -> b1 (conv(1/2)-bn-act) -> resnet_blocks -> out

        Returns
        -------
        out : (BS, hidden_dims[-1], last_h, last_w)
        """
        out = self.b1(x)
        out = self.resnet_blocks(out)
        return out


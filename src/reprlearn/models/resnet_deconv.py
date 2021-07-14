import torch.nn as nn
from typing import List


class ResidualDeconv(nn.Module):
    """A module that implements a single flow of residual operation for ResNet.
    Each conv layer uses kernel of size 3x3, stride=stride, and padding=1.
    First the input's (h,w) are shrinked by `stride`, then the num of channels
    is increased to out_c via subsequent conv operations.


    input ---> (bn-relu-convTranspose2d) -> z1 ---> (bn-relu-convTranspose2d) -> z2 ---> out
            |                                                                     ^
            |                                                                     |
            |                                                                     +
            ----------------->  (upsampling) --------------------------------------


    Parameters
    ----------
    stride : int
        Stride parameter of the first convTranspose layer. Use stride = 2 as a way to
        double the (h,w) of the input


    use_upsampling : bool
        Applies an upsampling and 1x1 conv to the input to adjust its (h,w) and
        n_channel (in_c) to be equal to the output's n_chhanel (out_c).
        Thus, it must be set to True whenever the input's num channel or (h,w) need to
        be adjusted in order to be added to the second conv's output (z2), ie:
        - in_c is different from out_c, or
        - `stride` != 1, or
        - same shape of input and output, but just want to add 1x1 conv operation
        to the input before adding it to the activation after the second conv.

    upsampling_type : str
        'nearest': use nearest neighbor unsampling (no extra parameters) followed by 1x1 conv
        'deconv': use convTranspose2d with `stride` to adjust both (h,w) and num of channels


    `forward(x)` returns
    -----------
    out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h/stride, in_w/stride)


    """

    def __init__(self, in_c, out_c,
                 *,
                 stride=2,
                 use_upsampling: bool = True,
                 upsampling_type: str = 'deconv',
                 norm_input: bool = True,
                 act_fn=nn.ReLU(inplace=True),
                 **kwargs
                 ):
        """
        To double the input's (h,w), ie. stride=2,
            use kernel_size=3, padding=1, stride=2, output_padding=1
        When the output needs bo have the same (h,w) as the input, ie. stride=1,
            use output_padding = 0
        """
        super().__init__()
        deconv_kwargs = {'kernel_size': 3, 'padding': 1}  # , 'output_padding':1}
        deconv_kwargs.update(kwargs)

        self.stride = stride
        self.outp = 1 if stride == 2 else 0
        self.use_upsampling = use_upsampling
        if in_c != out_c or stride > 1:
            assert self.use_upsampling == True, "Input needs to be adjusted in (h,w) and/or num channels. Set use_upsampling=True"
        self.upsampling = None
        if upsampling_type == 'nearest':
            self.upsampling = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=self.stride),
                nn.ConvTranspose2d(in_c, out_c, **deconv_kwargs, stride=1, output_padding=0)
            )
        elif upsampling_type == 'deconv':
            # Use 1x1 conv to do both channelwise and resolutionwise expansion
            self.upsampling = nn.ConvTranspose2d(in_c, out_c, **deconv_kwargs, stride=self.stride,
                                                 output_padding=self.outp)
        self.norm_input = norm_input

        self.bn1 = nn.BatchNorm2d(in_c)
        self.deconv1 = nn.ConvTranspose2d(in_c, out_c, **deconv_kwargs, stride=self.stride, output_padding=self.outp)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.deconv2 = nn.ConvTranspose2d(out_c, out_c, **deconv_kwargs, stride=1, output_padding=0)

        self.act_fn = act_fn

    def forward(self, x):
        """
        Returns
        -------
        out = model(x) returns a batch of tensors whose size is :
        (BS, out_c, in_h * stride, in_w *stride)

        """
        if self.norm_input:
            z = self.deconv1(self.act_fn(self.bn1(x)))
        else:
            z = self.deconv1(x)  # ; print(z.shape);breakpoint()
        z = self.deconv2(self.act_fn(self.bn2(z)))  # ; print(z.shape);breakpoint()

        if self.use_upsampling:
            x = self.upsampling(x)  # ; print(x.shape);breakpoint()
        z = z + x  # ; print(z.shape);breakpoint()

        return self.act_fn(z)


# Each of the subsequent blocks contain 2 residual operations, where the output channel
# is doubled and the resolution (ie. h,w) is halved
def get_resnet_deconv_block(
        in_c, out_c, *,
        n_residuals=2,
        first_block=False,
) -> List[nn.Module]:
    # First residual: In the first block, we don't apply the batchnorm because
    # the input is already processed with a batchnorm.

    norm_input = True
    if first_block:
        norm_input = False
    res0 = ResidualDeconv(in_c, out_c, stride=2, use_upsampling=True, norm_input=norm_input)

    block = [res0]
    # Add subsequence residuals
    for i in range(n_residuals - 1):
        block.append(ResidualDeconv(out_c, out_c, stride=1,
                                    use_upsampling=False, norm_input=True)
                     )
    return block


class ResNetDecoder(nn.Module):
    """x -> resnet_deconv_blocks -> out

    input -> resnet_deconv_blocks: nn.Sequential of resnet_deconv blocks
                Each block has 2 residual units: (bn-relu-conv, bn-relu-conv),
                except the first block, whose first residual unit doesn't apply (bn-relu)
    out : (BS, hidden_dims[-1]=in_channels, in_h, in_w)

    """

    def __init__(self,
                 nfs: List[int],
                 act_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.act_fn = act_fn

        blocks = []
        for i, (in_c, out_c) in enumerate(zip(nfs, nfs[1:])):
            is_first = (i == 0)
            blocks.extend(get_resnet_deconv_block(in_c, out_c, first_block=is_first))
        self.resnet_deconv_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x -> resnet_deconv_blocks -> out

        Returns
        -------
        out : (BS, hidden_dims[-1]=in_channels, in_h, in_w)
        """
        out = self.resnet_deconv_blocks(x)
        return out








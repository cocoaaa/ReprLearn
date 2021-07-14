from src.models.convnet import conv_block, conv_blocks


def test_conv_block():
    conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 2}
    b1 = conv_block(3, 10, **conv_kwargs)
    print(b1)
    for name, p in b1.named_parameters():
        print(name, p.shape)


def test_conv_blocks():
    cbs = conv_blocks(3, nf_list=[5, 6, 7])
    for name, p in cbs.named_parameters():
        print(name, p.shape)
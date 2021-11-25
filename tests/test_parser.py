"See: https://mkaz.blog/code/python-argparse-cookbook/"

import argparse
from reprlearn.models.plmodules.conv_fc_gan import ConvGenerator
from reprlearn.models.plmodules.conv_fc_gan import ConvFCGAN

def test_parser_basic():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--gpu_ids', action='store', type=str, nargs='*')
    my_parser.add_argument('--list_gpu_id', action='store', type=str, nargs=1)
    my_parser.add_argument('--int_gpu_id', action='store', type=str)

    args = my_parser.parse_args()
    print("---args---")
    print(args)

    print("gpu_ids: ", args.gpu_ids)
    print(','.join(args.gpu_ids))

    print("list_gpu_id: ", args.list_gpu_id)
    print(','.join(args.list_gpu_id))

    print("args.int_gpu_id: ", args.int_gpu_id)


def test_parser_boolean():
    # See: stackoverflow.com/a/31347222
    parser = argparse.ArgumentParser()
    parser.add_argument('--early_stop', dest='early_stop', action='store_true')
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false')
    parser.set_defaults(early_stop=True)

    for cmd in ['--early_stop', '--no_early_stop']:
        args = parser.parse_args(cmd.split())
        print('cli: ', cmd)
        print(args)


def test_parser_conflict_handler_1():
    parent_p = argparse.ArgumentParser()
    parent_p.add_argument('--dimz', type=int, default=10)

    parser = argparse.ArgumentParser(parents=[parent_p],
                                      conflict_handler='resolve')
    parser.add_argument('--dimz', type=int, default=20)

    args = parser.parse_args()
    print(args) # Namespace(dimz=20)


def test_parser_conflict_handler_2():
    parent_p = argparse.ArgumentParser()
    parent_p.add_argument('--dimz', type=int, default=10)

    parser = argparse.ArgumentParser(parents=[parent_p],
                                      conflict_handler='resolve')
    parser.add_argument('--dimz', type=int, default=20)

    args = parser.parse_args(['--dimz', '30'])
    print('type: ', type(parser))
    print(args) # Namespace(dimz=30)


def test_conv_gen_add_model_specific_args():
    parser = ConvGenerator.add_model_specific_args()
    args = parser.parse_args(['--latent_dim', '100'])
    print(args)


def test_conv_fc_gan_add_model_specific_args():
    parser = ConvFCGAN.add_model_specific_args()

    arg_str = '--latent_dim 100 --latent_emb_dim 32 --lr_g 1e-2 --lr_d 1e-1 -k 5'
    print('arg_str')

    args = parser.parse_args(arg_str.split())
    # or
    # args, _ = parser.parse_known_args(arg_str.split())

    print(args)


def test_conv_fc_gan_add_model_specific_args_with_parent():
    parent = argparse.ArgumentParser(add_help=False)  # add_help=False is important!
    parent.add_argument('--latent_dim', type=int, default=1)

    parser = ConvFCGAN.add_model_specific_args(parent)

    arg_str = '--latent_dim 100 --latent_emb_dim 32 --lr_g 1e-2 --lr_d 1e-1 -k 5'
    print('arg_str')

    args = parser.parse_args(arg_str.split())
    # or
    # args, _ = parser.parse_known_args(arg_str.split())

    print(args)


if __name__ == '__main__':
    # test_parser_basic()
    test_parser_boolean()
    # test_parser_conflict_handler_1()
    # test_parser_conflict_handler_2()
    # test_conv_gen_add_model_specific_args()
    # test_conv_fc_gan_add_model_specific_args()
    # test_conv_fc_gan_add_model_specific_args_with_parent()
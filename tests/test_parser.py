"See: https://mkaz.blog/code/python-argparse-cookbook/"

import argparse

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
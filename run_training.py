import argparse

from train import train_ijepa


parser = argparse.ArgumentParser(description='I-JEPA')
parser.add_argument('--dataset_path',
                    default='./data',
                    help='Path where datasets will be saved')
parser.add_argument('--dataset_name',
                    default='stl10',
                    help='Dataset name',
                    choices=['stl10'])
parser.add_argument('-save_model_dir',
                    default='./models',
                    help='Path where models')
parser.add_argument('--num_epochs',
                    default=100,
                    type=int,
                    help='Number of epochs for training')
parser.add_argument('-b',
                    '--batch_size',
                    default=256,
                    type=int,
                    help='Batch size')
parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float)
parser.add_argument('--fp16_precision',
                    action='store_true',
                    help='Whether to use 16-bit precision for GPU training')

parser.add_argument('--emb_dim',
                    default=768,
                    type=int,
                    help='Transofmer embedding dimm')
parser.add_argument('--log_every_n_steps',
                    default=200,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')
parser.add_argument('--update_gamma_after_step',
                    default=1,
                    type=int,
                    help='Update EMA gamma after this step')
parser.add_argument('--update_gamma_every_n_steps',
                    default=1,
                    type=int,
                    help='Update EMA gamma after this many steps')
parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Specify path to training_model.pth to resume training')


def main():
    args = parser.parse_args()
    train_ijepa(args)


if __name__ == "__main__":
    main()

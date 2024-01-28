import torch
from torch import nn
import argparse


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')

    # print(ckpt)
    print(ckpt.keys())

    # ckpt.keys() = dict_keys(['state_dict', 'config', 'epoch', 'date'])
    # save the checkpoint as a variable and return it
    ckpt = {'state_dict': ckpt['state_dict'],
            'config': ckpt['config'],
            'epoch': ckpt['epoch'],
            'date': ckpt['date'],
            }
    return ckpt

def save_ckpt(ckpt, ckpt_path):
    ckpt = {'state_dict': ckpt['state_dict'],
            'config': ckpt['config'],
            'epoch': ckpt['epoch'],
            'date': ckpt['date'],
            }
    torch.save(ckpt, ckpt_path)


if __name__ == '__main__':
    ckpt_path = 'geom_best.pt'
    ckpt = load_ckpt(ckpt_path)
    save_ckpt(ckpt, 'geom_best.pth')

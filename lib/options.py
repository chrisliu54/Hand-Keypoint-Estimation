import argparse
from functools import wraps
from os.path import join
from pprint import pprint

import yaml
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    help='to set the parameters')
parser.add_argument('--gpus', default=None, nargs='+', type=int,
                    help='the gpu used')

args = parser.parse_args()


def singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@singleton
class Config:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            parser = edict(yaml.load(f))
        parser.MISC.GPUS = args.gpus
        print('======CONFIGURATION START======')
        pprint(parser)
        print('======CONFIGURATION END======')
        self.parser = parser


config = Config(args.config).parser

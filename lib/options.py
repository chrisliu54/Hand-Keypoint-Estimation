import argparse
from functools import wraps
import os
from pprint import pprint

import yaml
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/train.yml', type=str,
                    help='to set the parameters')
parser.add_argument('--gpus', default=None, type=str,
                    help='the gpu used')
parser.add_argument('--task', default='', type=str,
                    help='supplementary info of the task, will be appended to project name')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

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
        parser.PROJ_NAME = parser.PROJ_NAME if len(args.task) == 0\
            else '{}-{}'.format(parser.PROJ_NAME, args.task)
        print('======CONFIGURATION START======')
        pprint(parser)
        print('======CONFIGURATION END======')
        self.parser = parser


config = Config(args.config).parser

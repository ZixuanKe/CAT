import sys,os,argparse,time
import numpy as np
import torch
import multiprocessing
import utils

def train_config(parser):
    parser=argparse.ArgumentParser(description='xxx')
    parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
    parser.add_argument('--experiment',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--approach',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--lr_min',default=1e-4,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--lr_factor',default=3,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--clipgrad',default=10000,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--lamb',default=0.75,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--lamb_probs',default=0.75,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--smax',default=400,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--smax_prob',default=1000,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--temperature_prob',default=5,type=int,required=False,help='(default=%(default)f)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
    parser.add_argument('--sbatch',default=64,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--nhid',default=2000,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--idrandom',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--classptask',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--pdrop1',default=0.2,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--pdrop2',default=0.5,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--filters',default=1,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--filter_num',default=100,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument("--num_class_femnist",default=62,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)f)')
    parser.add_argument('--dis_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--sim_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--data_size',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--similarity_detection',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--loss_type',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--lr_kt',default=0.025,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--nepochs_kt',default=300,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--lr_patience_kt',default=10,type=int,required=False,help='(default=%(default)f)')
    parser.add_argument("--n_head",default=1,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--model_weights',default=1,type=float,required=False,help='(default=%(default)d)')

    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = train_config(parser)

    args = parser.parse_args()
    return args

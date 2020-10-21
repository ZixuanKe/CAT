import sys,os,argparse,time


def set_config():
    # Arguments
    parser=argparse.ArgumentParser(description='xxx')
    parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
    parser.add_argument('--experiment',default='',type=str,required=True,choices=['alexmixemnistceleba','mlpmixemnistceleba',
                                                                                  'alexceleba','sentiment',
                                                                                  'alexmixceleba','mlpmixceleba',
                                                                                  'mixemnist','mnist','celeba',
                                                                                  'femnist','landmine','cifar10',
                                                                                  'cifar100','emnist','mnist2',
                                                                                  'pmnist','cifar','mixture'],help='(default=%(default)s)')
    parser.add_argument('--approach',default='',type=str,required=True,choices=['adam','random','sgd','sgd-frozen',
                                                                                'lwf','lfl','ewc','imm-mean',
                                                                                'progressive',
                                                                                'alexpathnet','mlppathnet',
                                                                                'imm-mode','sgd-restart',
                                                                                'joint','hat','hat-test'],help='(default=%(default)s)')
    parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
    parser.add_argument('--sbatch',default=0,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--nhid',default=2000,type=int,required=False,help='(default=%(default)d)')
    # parser.add_argument('--ntasks',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--idrandom',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--classptask',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--pdrop1',default=-1,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--pdrop2',default=-1,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--filters',default=1,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--filter_num',default=100,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument("--num_class_femnist",default=62,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)f)')
    parser.add_argument('--dis_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--sim_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--model_weights',default=1,type=float,required=False,help='(default=%(default)d)')
    args=parser.parse_args()


    return args
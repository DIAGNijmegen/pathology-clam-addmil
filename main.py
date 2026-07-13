from __future__ import print_function

import argparse
import pdb
import os
import math
import wandb

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, slide2vec_Dataset
# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score #Added to calculate F1-score

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # initialize wandb if requested
    if getattr(args, 'use_wandb', False):
        wandb_mode = 'offline' if getattr(args, 'wandb_offline', False) else 'online'
        tags = args.wandb_tags.split(',') if (getattr(args, 'wandb_tags', None) and len(args.wandb_tags) > 0) else None
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=args.wandb_name,
                   job_type=args.wandb_job_type,
                   tags=tags,
                   config=settings,
                   reinit=True,
                   mode=wandb_mode,
                   resume=args.wandb_resume)
    
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_val_f1 = [] #Added to calculate F1-score
    all_test_f1 = [] #Added to calculate F1-score
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results_val_dict, results_test_dict, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        #Added to calculate F1-score
        #####################################################
        #Validation 
        df_val_class = pd.DataFrame.from_dict(results_val_dict, orient='index')
        df_val_class = df_val_class[["slide_id", "pred_class", "label"]]

        #Test
        df_test_class = pd.DataFrame.from_dict(results_test_dict, orient='index')
        df_test_class = df_test_class[["slide_id", "pred_class", "label"]]

        #####################################################
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_val_f1.append(f1_score(df_val_class["label"].tolist(), df_val_class["pred_class"].tolist(), zero_division="warn"))
        all_test_f1.append(f1_score(df_test_class["label"].tolist(), df_test_class["pred_class"].tolist(), zero_division="warn"))

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results_test_dict)

        # log per-fold metrics to wandb
        if getattr(args, 'use_wandb', False):
            try:
                wandb.log({'fold': int(i),
                           'test_auc': float(test_auc),
                           'val_auc': float(val_auc),
                           'test_acc': float(test_acc),
                           'val_acc': float(val_acc),
                           'test_f1': float(all_test_f1[-1]),
                           'val_f1': float(all_val_f1[-1])}, step=int(i))
            except Exception:
                pass

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'test_f1' : all_test_f1, 'val_f1' : all_val_f1})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    # log aggregated metrics to wandb and finish
    if getattr(args, 'use_wandb', False):
        summary = {
            'test_auc_mean': float(np.mean(all_test_auc)),
            'val_auc_mean': float(np.mean(all_val_auc)),
            'test_acc_mean': float(np.mean(all_test_acc)),
            'val_acc_mean': float(np.mean(all_val_acc)),
            'test_f1_mean': float(np.mean(all_test_f1)),
            'val_f1_mean': float(np.mean(all_val_f1)),
            'num_folds': int(len(folds))
        }
        try:
            wandb.log(summary)
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'addmil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['cscc_vs_noncscc', 'cscc_vs_noncscc_slide2vec', 'mcscc_slide2vec', 'task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--data_label_csv_path', type=str, default=None, help='data label directory')    
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')  

parser.add_argument('--use_wandb', action='store_true', default=False, help='enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='histogenomics', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (team/user)')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
parser.add_argument('--wandb_job_type', type=str, default='train', help='wandb job type')
parser.add_argument('--wandb_tags', type=str, default=None, help='comma-separated tags for wandb run')
parser.add_argument('--wandb_offline', action='store_true', default=False, help='run wandb in offline mode')
parser.add_argument('--wandb_resume', type=str, default=None, help='resume wandb run by id or name')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

if args.checkpoint_path is not None:
    settings.update({'checkpoint_path': args.checkpoint_path})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
elif args.task == 'cscc_vs_noncscc':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                            data_dir= os.path.join(args.data_root_dir, 'features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'non-cscc':0, 'cscc':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'cscc_vs_noncscc_slide2vec':
    args.n_classes=2
    dataset = slide2vec_Dataset(csv_path = args.data_label_csv_path,
                            data_dir= os.path.join(args.data_root_dir, 'slide2vec'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'non-cscc':0, 'cscc':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'mcscc_slide2vec':
    args.n_classes=2
    dataset = slide2vec_Dataset(csv_path = args.data_label_csv_path,
                            data_dir= os.path.join(args.data_root_dir, 'slide2vec'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'control':0, 'case':1},
                            patient_strat=False,
                            ignore=[])

else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")



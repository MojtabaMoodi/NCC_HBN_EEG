# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
import sys
from pathlib import Path

# Ensure LaBraM directory is first on sys.path so "utils" and other local imports
# resolve to this package. Required when DataLoader workers re-execute this script
# (multiprocessing spawn); otherwise they may import the project-level EEG/utils package.
_script_dir = Path(__file__).resolve().parent
_script_dir_str = str(_script_dir)
try:
    sys.path.remove(_script_dir_str)
except ValueError:
    pass
sys.path.insert(0, _script_dir_str)

from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math

from collections import OrderedDict
from functools import partial
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate, evaluate_by_task_type, count_task_type_samples
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune

# Import dataset configuration from dedicated module
from dataset_config import (
    CUSTOM_DATASET_CONFIGS,
    get_dataset_config,
    get_dataset_type_and_params,
    get_data_path,
    DATA_PATHS,
)
from labram_dataset import collate_labram_with_participant_ids
from data_processing.eeg_dataset import (
    compute_class_weights_from_train_hdf5,
    get_dataloader_multiprocessing_kwargs,
    get_recommended_num_workers,
)

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--aggregate_by_participant', type=str, default=None, metavar='METHOD',
                        help="Participant-level aggregation: 'majority_vote' for classification (same as CNN). Reports segment-level and participant-level (mean prob + majority vote) metrics.")
    parser.add_argument('--early_stopping_patience', default=0, type=int,
                        help='Stop after this many consecutive epochs with no validation improvement (0 = disabled). '
                             'Monitors val accuracy (classification) or val MAE in years (regression). '
                             'Best checkpoint is still saved when the metric improves; use checkpoint-best.pth after stop.')
    parser.add_argument('--early_stopping_min_delta', default=0.0, type=float,
                        help='Minimum relative improvement on the monitored val metric to reset patience '
                             '(classification: val accuracy must increase by more than this; '
                             'regression: val MAE must decrease by more than this, in years if MAE is in years).')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV | AGE | GENDER | COMBINED | MULTI_OUTPUT | AGE_CV_GENDER_STRATIFIED | GENDER_CROSS_TASK_ACTIVE_TO_PASSIVE | etc.')
    
    parser.add_argument('--segment_length', default='1s', type=str, choices=['1s', '2s', '4s'],
                        help='Segment length: 1s or 4s (only for custom datasets)')
    
    parser.add_argument('--data_path', default=None, type=str,
                        help='HDF5 data directory (train/val/test multipart: eeg_data_{train,val,test}_{1s,2s,4s}_part*.h5). Required if EEG_HDF5_DIR is not set.')
    # Class imbalance (binary / gender): same as CNN — inverse-frequency weight for positive class.
    parser.add_argument('--class_weight_power', default=1.0, type=float,
                        help='Exponent for class weights from train data (default 1.0). Use >1 (e.g. 1.5) for stronger minority weighting. Only for binary (gender).')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    # Detect multi-output mode from dataset type
    is_multi_output = args.dataset.startswith('multi_output') if hasattr(args, 'dataset') else False
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        multi_output=is_multi_output,  # Enable multi-output mode
        prediction_type=getattr(args, 'prediction_type', 'classification'),
    )

    return model


# Constants for our custom datasets
CHANNEL_NAMES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
    'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
    'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
    'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
]

def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("path/to/TUEV")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    
    # Handle our custom datasets using direct prepare_custom_dataset calls
    elif args.dataset in CUSTOM_DATASET_CONFIGS:
        # Get dataset configuration dynamically
        config = get_dataset_config(args.dataset)
        args.nb_classes = config['nb_classes']
        metrics = config['metrics']
        
        # Get channel names (all our custom datasets use standard 10-20 channels)
        ch_names = CHANNEL_NAMES
        
        # Determine data path
        if args.data_path is not None:
            data_path = args.data_path
        else:
            # Auto-detect based on segment length for custom datasets
            if hasattr(args, 'segment_length'):
                data_path = get_data_path(args.segment_length)
            else:
                # Default to 1s segments
                data_path = get_data_path('1s')
        
        # Extract dataset type and parameters from dataset name
        dataset_type, kwargs = get_dataset_type_and_params(args.dataset)
        args.prediction_type = 'regression' if dataset_type == 'age_regression' else 'classification'
        
        # Add segment_length to kwargs for HDF5 dataset preparation
        if hasattr(args, 'segment_length'):
            kwargs['segment_length'] = args.segment_length
        
        # Call prepare_custom_dataset directly
        result = utils.prepare_custom_dataset(dataset_type, data_path, **kwargs)
        
        # Handle different return types
        if isinstance(result, tuple) and len(result) == 3:
            # Basic datasets return (train, test, val)
            train_dataset, test_dataset, val_dataset = result
            cv_folds = None  # Not a CV dataset
            if dataset_type == 'age_regression':
                for attr in ('age_min', 'age_max'):
                    if not hasattr(train_dataset, attr):
                        raise RuntimeError(
                            f"Age regression requires LaBraMEEGDataset.{attr} from prepare_labram_dataset."
                        )
                args.age_min = float(train_dataset.age_min)
                args.age_max = float(train_dataset.age_max)
        elif isinstance(result, list):
            # Cross-validation datasets return list of folds
            cv_folds = result  # Store all folds for later processing
            train_dataset, val_dataset, test_dataset = result[0]  # Use first for setup
        else:
            raise ValueError(f"Unexpected return type from prepare_custom_dataset: {type(result)}")
    else:
        cv_folds = None
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Check if this is a CV experiment and store it in args
    args.is_cross_validation = cv_folds is not None
    args.cv_folds = cv_folds
    
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def train_single_fold(args, ds_init, cv_fold_idx=None, cv_datasets=None):
    """
    Train a single fold of cross-validation or a regular experiment.
    
    Args:
        args: Command line arguments
        ds_init: DeepSpeed initialization function (if enabled)
        cv_fold_idx: Index of the current fold (None for regular experiments)
        cv_datasets: List of CV fold datasets (None for regular experiments)
    
    Returns:
        Dictionary with metrics for this fold
    """
    # Eval-only: run single-process to avoid requiring MASTER_ADDR/MASTER_PORT
    if args.eval:
        args.distributed = False
        args.rank = 0
        args.gpu = 0
        args.world_size = 1
    else:
        utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    
    # Handle CV fold-specific output directory
    original_output_dir = getattr(args, 'output_dir', './outputs')
    if cv_fold_idx is not None:
        args.output_dir = os.path.join(original_output_dir, f"fold_{cv_fold_idx + 1}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training fold {cv_fold_idx + 1} of {len(cv_datasets)}")
    else:
        # For non-CV experiments, create output directory if it doesn't exist
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prefer GPU when requested and available; no hardcoded device indices
    if (args.device == 'cuda' or (isinstance(args.device, str) and args.device.startswith('cuda'))):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print("WARNING: CUDA not available; using CPU. Install PyTorch with CUDA for GPU training.")
    else:
        device = torch.device(args.device)
    if device.type == 'cuda':
        n_gpu = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"Using device: {device} ({n_gpu} GPU(s): {gpu_names})")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    
    # Load datasets - use CV fold if provided
    if cv_datasets is not None and cv_fold_idx is not None:
        train_dataset, val_dataset, test_dataset = cv_datasets[cv_fold_idx]
        # Get channel names and metrics from the base configuration
        dataset_config = get_dataset_config(args.dataset)
        ch_names = CHANNEL_NAMES  # All custom datasets use the same channels
        metrics = dataset_config['metrics']
        dataset_type_cv, _ = get_dataset_type_and_params(args.dataset)
        args.prediction_type = 'regression' if dataset_type_cv == 'age_regression' else 'classification'
        if dataset_type_cv == 'age_regression':
            if not hasattr(train_dataset, 'age_min') or not hasattr(train_dataset, 'age_max'):
                raise RuntimeError("CV age regression fold datasets must expose age_min/age_max on the train wrapper.")
            args.age_min = float(train_dataset.age_min)
            args.age_max = float(train_dataset.age_max)
    else:
        dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
        train_dataset, val_dataset, test_dataset = dataset_train, dataset_val, dataset_test

    is_reg_task = getattr(args, 'prediction_type', 'classification') == 'regression'
    cls_is_binary = (args.nb_classes == 1) and (not is_reg_task)


    # Note: IterableDataset doesn't support samplers (DistributedSampler, RandomSampler, etc.)
    # DataLoader will iterate through the dataset directly without a sampler
    # For distributed training with IterableDataset, sharding should be handled within the dataset
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    print(f"Using IterableDataset - DataLoader will iterate directly (no sampler)")
    print(f"Distributed training: {num_tasks} tasks, rank {global_rank}")
    
    # Set samplers to None for IterableDataset
    sampler_train = None
    sampler_val = None
    sampler_test = None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # LaBraMEEGDataset yields (eeg, label, participant_id); use custom collate so batch is (eeg, label, pids). Training only uses (eeg, label).
    use_labram_collate = args.dataset in CUSTOM_DATASET_CONFIGS
    if use_labram_collate and is_reg_task:
        labram_collate = partial(collate_labram_with_participant_ids, label_mode='regression')
    elif use_labram_collate:
        labram_collate = collate_labram_with_participant_ids
    else:
        labram_collate = None
    val_test_collate = labram_collate

    # num_workers: for custom HDF5 datasets use segment-length-aware recommendation (avoids 1s hang with fork + many workers).
    if args.dataset in CUSTOM_DATASET_CONFIGS:
        segment_length = getattr(args, "segment_length", "1s")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if segment_length == "1s":
            num_workers = 0
            if utils.is_main_process():
                print("DataLoader num_workers: 0 (1s segments: single process to avoid each worker building HDF5 cache; first batch may take 10-30 min)")
        else:
            num_workers = get_recommended_num_workers(segment_length, num_gpus)
            if utils.is_main_process():
                print(f"DataLoader num_workers: {num_workers} (segment_length={segment_length}, num_gpus={num_gpus})")
    else:
        num_workers = args.num_workers

    # Multiprocessing kwargs for HDF5/IterableDataset: spawn context to avoid hang (same as CNN).
    loader_mp_kwargs = get_dataloader_multiprocessing_kwargs(num_workers, prefetch_factor=2)

    # IterableDataset doesn't support samplers - DataLoader iterates directly
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,  # No sampler for IterableDataset
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=labram_collate,
        **loader_mp_kwargs,
    )
    if val_dataset is not None:
        data_loader_val = torch.utils.data.DataLoader(
            val_dataset,  # No sampler for IterableDataset
            batch_size=int(1.5 * args.batch_size),
            num_workers=num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=val_test_collate,
            **loader_mp_kwargs,
        )
        if type(test_dataset) == list:
            data_loader_test = [
                torch.utils.data.DataLoader(
                    dataset,  # No sampler for IterableDataset
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False,
                    collate_fn=val_test_collate,
                    **loader_mp_kwargs,
                )
                for dataset in test_dataset
            ]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                test_dataset,  # No sampler for IterableDataset
                batch_size=int(1.5 * args.batch_size),
                num_workers=num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                collate_fn=val_test_collate,
                **loader_mp_kwargs,
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location=device, check_hash=True)
        else:
            # Pretrained checkpoint may contain numpy/torch types; use weights_only=False for compatibility (trusted source).
            checkpoint = torch.load(args.finetune, map_location=device, weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        # Remove head weights that don't match (single or multi-output)
        head_keys_to_remove = []
        if model.multi_output:
            for k in ['gender_head.weight', 'gender_head.bias', 'age_head.weight', 'age_head.bias']:
                if k in checkpoint_model and (k not in state_dict or checkpoint_model[k].shape != state_dict[k].shape):
                    head_keys_to_remove.append(k)
        else:
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    head_keys_to_remove.append(k)
        
        for k in head_keys_to_remove:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    # Use all visible GPUs when not in distributed mode
    if not args.distributed and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module
        print(f"Wrapped model in DataParallel across {torch.cuda.device_count()} GPU(s).")
    else:
        model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model_without_ddp,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    # Note: IterableDataset doesn't support __len__(), so we can't calculate exact steps per epoch
    # We'll use a large placeholder value for LR scheduling, and let the training loop naturally
    # exhaust the iterator. The actual steps will be determined during the first epoch.
    # Using a large value (100000) ensures the LR schedule covers all possible steps.
    num_training_steps_per_epoch = 100000  # Large placeholder for LR scheduling - iterator will naturally exhaust
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = Unknown (IterableDataset)")
    print("Number of training steps per epoch = Determined during training (IterableDataset)")
    print("Note: LR schedule uses placeholder value - actual steps determined by iterator exhaustion")

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if getattr(args, 'prediction_type', 'classification') == 'regression':
        criterion = torch.nn.MSELoss()
        print("criterion = MSELoss() (age regression, targets in [0, 1])")
    elif args.nb_classes == 1:
        # Binary (gender): use pos_weight from train class counts to prevent collapse to majority class (same idea as CNN).
        bce_pos_weight = None
        if args.dataset in CUSTOM_DATASET_CONFIGS:
            hdf5_dir = args.data_path if args.data_path is not None else get_data_path(getattr(args, 'segment_length', '1s'))
            segment_length_str = getattr(args, 'segment_length', '1s')
            dataset_type, ds_kwargs = get_dataset_type_and_params(args.dataset)
            if dataset_type == 'gender':
                task_type = ds_kwargs.get('train_task_type', ds_kwargs.get('task_type', 'both'))
                weight_power = getattr(args, 'class_weight_power', 1.0)
                try:
                    weights = compute_class_weights_from_train_hdf5(
                        hdf5_dir, segment_length_str, 'gender',
                        task_type=task_type, weight_power=weight_power,
                    )
                    if len(weights) == 2 and weights[0] > 0:
                        bce_pos_weight = weights[1] / weights[0]
                        print(f"Binary class weights from train data (gender): {[round(w, 4) for w in weights]} -> BCE pos_weight={bce_pos_weight:.4f}")
                except Exception as e:
                    print(f"Could not compute class weights for BCE: {e}. Using unweighted BCE.")
        if bce_pos_weight is not None:
            pos_t = torch.tensor([bce_pos_weight], dtype=torch.float32, device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_t)
            print("Using BCEWithLogitsLoss(pos_weight=...) to reduce majority-class collapse")
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    if args.eval:
        aggregate = getattr(args, 'aggregate_by_participant', None)
        balanced_accuracy = []
        accuracy = []
        # Wrap single DataLoader in a list so we iterate over loaders, not over batches
        test_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]
        for data_loader in test_loaders:
            test_stats = evaluate(
                data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                is_binary=cls_is_binary,
                aggregate_by_participant=aggregate,
                is_regression=is_reg_task,
                age_min=getattr(args, 'age_min', None),
                age_max=getattr(args, 'age_max', None),
            )
            if is_reg_task:
                accuracy.append(test_stats.get('mae', float('nan')))
                balanced_accuracy.append(test_stats.get('r2', float('nan')))
            else:
                accuracy.append(test_stats['accuracy'])
                balanced_accuracy.append(test_stats.get('balanced_accuracy', 0.0))
            if test_stats.get('segment_metrics') and test_stats.get('participant_metrics'):
                seg_acc = test_stats['segment_metrics'].get('accuracy')
                part_mp = test_stats.get('participant_mean_prob_metrics') or {}
                part_mv = test_stats.get('participant_metrics') or {}
                part_mp_acc = part_mp.get('accuracy')
                part_mv_acc = part_mv.get('accuracy')
                if seg_acc is not None:
                    _mp = f'{part_mp_acc:.4f}' if part_mp_acc is not None else 'N/A'
                    _mv = f'{part_mv_acc:.4f}' if part_mv_acc is not None else 'N/A'
                    print(f'  Test segment-level: {seg_acc:.4f} | participant (mean prob): {_mp} | participant (majority vote): {_mv}')
        if is_reg_task:
            print(f"======Test MAE (years): mean={np.nanmean(accuracy)} std={np.nanstd(accuracy)}, R²: mean={np.nanmean(balanced_accuracy)} std={np.nanstd(balanced_accuracy)}")
        else:
            print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    best_val_mae = float('inf')
    best_test_mae = float('nan')
    
    # Count task types for validation and test sets (once, before training)
    val_task_counts = None
    test_task_counts = None
    if data_loader_val is not None:
        print(f"\n{'='*60}")
        print("Counting task types in validation and test sets...")
        print(f"{'='*60}")
        val_task_counts = count_task_type_samples(data_loader_val)
        print(f"  Validation: {val_task_counts['active']:,} active, {val_task_counts['passive']:,} passive samples")
        if type(data_loader_test) == list:
            # Multiple test sets - count first one
            test_task_counts = count_task_type_samples(data_loader_test[0])
            print(f"  Test (first set): {test_task_counts['active']:,} active, {test_task_counts['passive']:,} passive samples")
        else:
            test_task_counts = count_task_type_samples(data_loader_test)
            print(f"  Test: {test_task_counts['active']:,} active, {test_task_counts['passive']:,} passive samples")
        print(f"{'='*60}\n")
    
    training_unstable = False
    early_stop_collapse = False
    early_stop_patience = False
    early_stopping_epochs_without_improvement = 0
    try:
        for epoch in range(args.start_epoch, args.epochs):
            if early_stop_collapse or early_stop_patience:
                break
            if args.distributed:
                # Note: IterableDataset doesn't use samplers, so no set_epoch needed
                # For IterableDataset, each epoch naturally starts from the beginning
                pass
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                ch_names=ch_names, is_binary=cls_is_binary, is_regression=is_reg_task,
            )
        
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
                
            if data_loader_val is not None:
                val_stats = evaluate(
                    data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                    is_binary=cls_is_binary,
                    aggregate_by_participant=getattr(args, 'aggregate_by_participant', None),
                    is_regression=is_reg_task,
                    age_min=getattr(args, 'age_min', None),
                    age_max=getattr(args, 'age_max', None),
                )
                test_stats = evaluate(
                    data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                    is_binary=cls_is_binary,
                    aggregate_by_participant=getattr(args, 'aggregate_by_participant', None),
                    is_regression=is_reg_task,
                    age_min=getattr(args, 'age_min', None),
                    age_max=getattr(args, 'age_max', None),
                )
                
                # Evaluate by task type (active/passive) every 5 epochs or on last epoch
                task_type_results = None
                if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                    if args.dataset in CUSTOM_DATASET_CONFIGS:
                        try:
                            # Get dataset type and data path
                            dataset_type, _ = get_dataset_type_and_params(args.dataset)
                            if args.data_path is not None:
                                hdf5_dir = args.data_path
                            else:
                                hdf5_dir = get_data_path(getattr(args, 'segment_length', '1s'))
                            segment_length = getattr(args, 'segment_length', '1s')
                            
                            task_type_results = evaluate_by_task_type(
                                model, device, dataset_type, hdf5_dir, segment_length,
                                ch_names=ch_names, metrics=metrics, is_binary=cls_is_binary,
                                random_seed=args.seed,
                                is_regression=is_reg_task,
                                age_min=getattr(args, 'age_min', None),
                                age_max=getattr(args, 'age_max', None),
                            )
                            
                            # Print task-type-specific results
                            for task_type, results in task_type_results.items():
                                num_samples = results.get('num_samples', 0)
                                if is_reg_task:
                                    mae_y = results.get('mae', float('nan'))
                                    print(f"  {task_type.upper()} Test MAE (years): {mae_y:.4f} (n={num_samples:,})")
                                else:
                                    acc = results.get('accuracy', 0.0) * 100
                                    print(f"  {task_type.upper()} Test Acc: {acc:.4f}% (n={num_samples:,})")
                        except Exception as e:
                            print(f"  Warning: Could not evaluate by task type: {e}")
                
                # Print epoch summary in a clear format (matching CNN format)
                train_loss = train_stats.get('loss', 0.0)
                val_loss = val_stats.get('loss', 0.0)
                if is_reg_task:
                    train_score = train_stats.get('class_acc', 0.0)
                    val_mae_ep = val_stats.get('mae', float('nan'))
                    print(f"  → Epoch {epoch+1:3d}/{args.epochs} | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train score (1/(1+norm_MAE)): {train_score:.4f} | Val MAE (years): {val_mae_ep:.4f} | "
                          f"LR: {current_lr:.6f}")
                else:
                    train_acc = train_stats.get('class_acc', 0.0) * 100  # Convert to percentage
                    val_acc = val_stats.get('accuracy', 0.0) * 100  # Convert to percentage
                    print(f"  → Epoch {epoch+1:3d}/{args.epochs} | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}% | "
                          f"LR: {current_lr:.6f}")

                old_best_val_acc = max_accuracy
                old_best_val_mae = best_val_mae
                
                if is_reg_task:
                    cur_val_mae = val_stats.get('mae', float('inf'))
                    if cur_val_mae < best_val_mae:
                        best_val_mae = cur_val_mae
                        best_test_mae = test_stats.get('mae', float('nan'))
                        torch.cuda.empty_cache()
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                elif max_accuracy < val_stats["accuracy"]:
                    max_accuracy = val_stats["accuracy"]
                    max_accuracy_test = test_stats["accuracy"]
                    
                    # Clear GPU cache before saving to avoid OOM
                    torch.cuda.empty_cache()
                    
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
    
                if is_reg_task:
                    print(f'Best val MAE (years): {best_val_mae:.4f}, test MAE at best val: {best_test_mae:.4f}')
                else:
                    print(f'Max accuracy val: {max_accuracy * 100:.2f}%, max accuracy test: {max_accuracy_test * 100:.2f}%')
                # Early stop on training collapse (classification only)
                if not is_reg_task:
                    nb_classes = max(1, args.nb_classes)
                    random_baseline = 1.0 / nb_classes
                    min_good_accuracy = random_baseline + 0.30
                    collapse_threshold = random_baseline + 0.15
                    if max_accuracy > min_good_accuracy and val_stats["accuracy"] < collapse_threshold:
                        best_ckpt_path = Path(args.output_dir) / utils.CHECKPOINT_BEST_FILENAME
                        print(f"\n⚠️  Training collapse detected: val accuracy dropped from {max_accuracy*100:.1f}% to {val_stats['accuracy']*100:.1f}%. "
                              f"Stopping early. Use {best_ckpt_path} (already saved) for evaluation.")
                        early_stop_collapse = True
                        break  # exit loop from inside the for-body (avoids lint error on break outside loop)
                # Patience early stopping (optional): no improvement on validation for N consecutive epochs
                esp = getattr(args, 'early_stopping_patience', 0) or 0
                if esp > 0:
                    min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0) or 0.0)
                    if is_reg_task:
                        cur_m = float(val_stats.get('mae', float('inf')))
                        if not math.isfinite(cur_m):
                            raise ValueError(
                                f'early_stopping_patience requires finite val MAE; got {val_stats.get("mae")!r}.'
                            )
                        metric_improved = (old_best_val_mae - cur_m) > min_delta
                    else:
                        cur_a = float(val_stats.get('accuracy', 0.0))
                        metric_improved = (cur_a - old_best_val_acc) > min_delta
                    if metric_improved:
                        early_stopping_epochs_without_improvement = 0
                    else:
                        early_stopping_epochs_without_improvement += 1
                    if early_stopping_epochs_without_improvement >= esp:
                        best_ckpt_path = Path(args.output_dir) / utils.CHECKPOINT_BEST_FILENAME
                        print(
                            f"\nEarly stopping: validation did not improve for {esp} consecutive epoch(s) "
                            f"(min_delta={min_delta}). Best checkpoint: {best_ckpt_path}"
                        )
                        early_stop_patience = True
                        break
                if test_stats.get('segment_metrics'):
                    if is_reg_task and test_stats.get('participant_median_metrics') and test_stats.get('participant_mean_metrics'):
                        seg_mae = test_stats['segment_metrics'].get('mae')
                        pm = test_stats.get('participant_median_metrics') or {}
                        pmean = test_stats.get('participant_mean_metrics') or {}
                        pm_mae = pm.get('mae')
                        pmean_mae = pmean.get('mae')
                        if seg_mae is not None:
                            _med = f'{pm_mae:.4f}' if pm_mae is not None else 'N/A'
                            _mean = f'{pmean_mae:.4f}' if pmean_mae is not None else 'N/A'
                            print(f'  Test segment MAE: {seg_mae:.4f} | participant median MAE: {_med} | participant mean MAE: {_mean}')
                    elif test_stats.get('participant_metrics'):
                        seg_acc = test_stats['segment_metrics'].get('accuracy')
                        part_mp = test_stats.get('participant_mean_prob_metrics') or {}
                        part_mv = test_stats.get('participant_metrics') or {}
                        part_mp_acc = part_mp.get('accuracy')
                        part_mv_acc = part_mv.get('accuracy')
                        if seg_acc is not None:
                            _mp = f'{part_mp_acc:.4f}' if part_mp_acc is not None else 'N/A'
                            _mv = f'{part_mv_acc:.4f}' if part_mv_acc is not None else 'N/A'
                            print(f'  Test segment-level: {seg_acc:.4f} | participant (mean prob): {_mp} | participant (majority vote): {_mv}')
                if log_writer is not None:
                    for key, value in val_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="val", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="val", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="val", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="val", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="val", step=epoch)
                        elif key == 'mae':
                            log_writer.update(mae=value, head="val", step=epoch)
                        elif key == 'mse':
                            log_writer.update(mse=value, head="val", step=epoch)
                        elif key == 'rmse':
                            log_writer.update(rmse=value, head="val", step=epoch)
                        elif key == 'r2':
                            log_writer.update(r2=value, head="val", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="val", step=epoch)
                    for key, value in test_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="test", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="test", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="test", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="test", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="test", step=epoch)
                        elif key == 'mae':
                            log_writer.update(mae=value, head="test", step=epoch)
                        elif key == 'mse':
                            log_writer.update(mse=value, head="test", step=epoch)
                        elif key == 'rmse':
                            log_writer.update(rmse=value, head="test", step=epoch)
                        elif key == 'r2':
                            log_writer.update(r2=value, head="test", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="test", step=epoch)
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
                if test_stats.get('segment_metrics') is not None:
                    log_stats['test_segment_metrics'] = test_stats['segment_metrics']
                if test_stats.get('participant_mean_prob_metrics') is not None:
                    log_stats['test_participant_mean_prob_metrics'] = test_stats['participant_mean_prob_metrics']
                if test_stats.get('participant_median_metrics') is not None:
                    log_stats['test_participant_median_metrics'] = test_stats['participant_median_metrics']
                if test_stats.get('participant_mean_metrics') is not None:
                    log_stats['test_participant_mean_metrics'] = test_stats['participant_mean_metrics']
    
                # Add task type counts if available (from first epoch)
                if epoch == 0:
                    if train_stats.get('task_type_counts'):
                        log_stats['train_task_type_counts'] = train_stats['task_type_counts']
                    if val_task_counts:
                        log_stats['val_task_type_counts'] = val_task_counts
                    if test_task_counts:
                        log_stats['test_task_type_counts'] = test_task_counts
                
                # Add task-type-specific metrics if available
                if task_type_results is not None:
                    if is_reg_task:
                        log_stats['task_type_metrics'] = {
                            task_type: {
                                'mae': results.get('mae', float('nan')),
                                'mse': results.get('mse', float('nan')),
                                'rmse': results.get('rmse', float('nan')),
                                'r2': results.get('r2', float('nan')),
                                'num_samples': results.get('num_samples', 0),
                            }
                            for task_type, results in task_type_results.items()
                        }
                    else:
                        log_stats['task_type_metrics'] = {
                            task_type: {
                                'accuracy': results.get('accuracy', 0.0),
                                'f1_weighted': results.get('f1_weighted', 0.0),
                                'balanced_accuracy': results.get('balanced_accuracy', 0.0),
                                'num_samples': results.get('num_samples', 0),
                            }
                            for task_type, results in task_type_results.items()
                        }
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
    
            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    except RuntimeError as e:
        training_unstable = True
        print(f"\n{e}")
        print("Loading best checkpoint for final evaluation (see below).")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Final evaluation by task type (active/passive) after training
    final_task_type_results = None
    if args.dataset in CUSTOM_DATASET_CONFIGS:
        try:
            # Load best model for final evaluation
            best_checkpoint_path = Path(args.output_dir) / utils.CHECKPOINT_BEST_FILENAME
            if best_checkpoint_path.exists():
                print(f"\nLoading best model from {best_checkpoint_path} for final evaluation...")
                # Our own checkpoint may contain optimizer state; use weights_only=False for compatibility.
                checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                model_without_ddp.load_state_dict(checkpoint['model'])
            
            # Get dataset type and data path
            dataset_type, _ = get_dataset_type_and_params(args.dataset)
            if args.data_path is not None:
                hdf5_dir = args.data_path
            else:
                hdf5_dir = get_data_path(getattr(args, 'segment_length', '1s'))
            segment_length = getattr(args, 'segment_length', '1s')
            
            print(f"\n{'='*80}")
            print("Final evaluation by task type (active/passive)")
            print(f"{'='*80}")
            final_task_type_results = evaluate_by_task_type(
                model, device, dataset_type, hdf5_dir, segment_length,
                ch_names=ch_names, metrics=metrics, is_binary=cls_is_binary,
                random_seed=args.seed,
                is_regression=is_reg_task,
                age_min=getattr(args, 'age_min', None),
                age_max=getattr(args, 'age_max', None),
            )
            
            # Print final task-type-specific results
            print(f"\n{'='*80}")
            print("Final Task-Type-Specific Results:")
            print(f"{'='*80}")
            for task_type, results in final_task_type_results.items():
                num_samples = results.get('num_samples', 0)
                if is_reg_task:
                    mae_y = results.get('mae', float('nan'))
                    r2v = results.get('r2', float('nan'))
                    print(f"  {task_type.upper()} tasks: MAE (years) = {mae_y:.4f}, R² = {r2v:.4f} (n={num_samples:,})")
                else:
                    acc = results.get('accuracy', 0.0) * 100
                    f1 = results.get('f1_weighted', 0.0)
                    print(f"  {task_type.upper()} tasks: Accuracy = {acc:.4f}%, F1-Score = {f1:.4f} (n={num_samples:,})")
        except Exception as e:
            print(f"Warning: Could not perform final task-type evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final model even if checkpoint saving was disabled
    if args.output_dir and not args.save_ckpt:
        # Save final epoch model as best checkpoint for easy loading
        final_checkpoint_path = Path(args.output_dir) / utils.CHECKPOINT_BEST_FILENAME
        print(f"\nSaving final model to {final_checkpoint_path}")
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs - 1,
            'args': args,
        }, final_checkpoint_path)
    
    # Return final metrics including task-type-specific results
    if is_reg_task:
        final_metrics = {
            'best_val_mae': best_val_mae,
            'test_mae_at_best_val': best_test_mae,
        }
    else:
        final_metrics = {
            'val_accuracy': max_accuracy,
            'test_accuracy': max_accuracy_test,
        }
    
    # Add task-type-specific metrics if available
    if final_task_type_results is not None:
        if is_reg_task:
            final_metrics['task_type_metrics'] = {
                task_type: {
                    'mae': results.get('mae', float('nan')),
                    'mse': results.get('mse', float('nan')),
                    'rmse': results.get('rmse', float('nan')),
                    'r2': results.get('r2', float('nan')),
                    'num_samples': results.get('num_samples', 0),
                }
                for task_type, results in final_task_type_results.items()
            }
        else:
            final_metrics['task_type_metrics'] = {
                task_type: {
                    'accuracy': results.get('accuracy', 0.0),
                    'f1_weighted': results.get('f1_weighted', 0.0),
                    'balanced_accuracy': results.get('balanced_accuracy', 0.0),
                    'num_samples': results.get('num_samples', 0),
                }
                for task_type, results in final_task_type_results.items()
            }
    
    # Add task type counts to final metrics
    if val_task_counts:
        final_metrics['val_task_type_counts'] = val_task_counts
    if test_task_counts:
        final_metrics['test_task_type_counts'] = test_task_counts
    
    # Restore original output directory
    args.output_dir = original_output_dir
    
    return final_metrics


def main(args, ds_init):
    """Main function that handles both regular and CV experiments."""
    print("LaBraM fine-tuning starting...", flush=True)

    # Pre-load datasets to check if this is a CV experiment
    if args.dataset in CUSTOM_DATASET_CONFIGS:
        # Get dataset configuration dynamically
        config = get_dataset_config(args.dataset)
        args.nb_classes = config['nb_classes']
        
        # Determine data path
        if args.data_path is not None:
            data_path = args.data_path
        else:
            if hasattr(args, 'segment_length'):
                data_path = get_data_path(args.segment_length)
            else:
                data_path = get_data_path('1s')
        
        # Extract dataset type and parameters from dataset name
        dataset_type, kwargs = get_dataset_type_and_params(args.dataset)
        if hasattr(args, 'segment_length'):
            kwargs['segment_length'] = args.segment_length

        # Check if this is a CV experiment by calling prepare_custom_dataset early
        # This will load all folds into memory, so we know if it's CV
        print("Loading dataset configuration (this may take a moment)...", flush=True)
        try:
            result = utils.prepare_custom_dataset(dataset_type, data_path, **kwargs)
            
            # Determine if CV
            if isinstance(result, list):
                args.is_cross_validation = True
                args.cv_folds = result
                print(f"Detected CV experiment: {len(result)} folds")
            else:
                args.is_cross_validation = False
                args.cv_folds = None
        except Exception as e:
            print(f"Warning: Could not preload dataset: {e}")
            args.is_cross_validation = False
            args.cv_folds = None
    else:
        args.is_cross_validation = False
        args.cv_folds = None
    
    # Check if this is a CV experiment
    if args.is_cross_validation and args.cv_folds:
        print(f"Running {len(args.cv_folds)}-fold cross-validation...")
        
        # Run each fold
        all_metrics = []
        for fold_idx in range(len(args.cv_folds)):
            print(f"\n{'='*80}")
            print(f"Training fold {fold_idx + 1}/{len(args.cv_folds)}")
            print(f"{'='*80}")
            
            metrics = train_single_fold(args, ds_init, cv_fold_idx=fold_idx, cv_datasets=args.cv_folds)
            all_metrics.append(metrics)
        
        # Calculate and print average metrics
        avg_val_acc = np.mean([m['val_accuracy'] for m in all_metrics])
        avg_test_acc = np.mean([m['test_accuracy'] for m in all_metrics])
        std_val_acc = np.std([m['val_accuracy'] for m in all_metrics])
        std_test_acc = np.std([m['test_accuracy'] for m in all_metrics])
        
        print(f"\n{'='*80}")
        print("Cross-Validation Summary")
        print(f"{'='*80}")
        print(f"Average Validation Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%")
        print(f"{'='*80}")
        
    else:
        # Regular experiment - single fold
        train_single_fold(args, ds_init)


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)

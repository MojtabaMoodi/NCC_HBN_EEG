#!/usr/bin/env python3
"""
Main script for running saliency analysis on EEG models (saliency maps + topography).

Single model: specify checkpoint and output locations via args or env (CHECKPOINT, OUTPUT_DIR).
Batch: use BEST_MODELS_CONFIG in code, or --checkpoint_list <file> to specify checkpoint paths.

Usage (Single Model):
    python main.py --checkpoint <path> --target_type <gender|age> --segment_length <1s|2s|4s> [options]
    # Or: export CHECKPOINT=/path/to/best.pth OUTPUT_DIR=/path/to/results
    python main.py --target_type gender --segment_length 4s

Usage (Batch - config in code):
    python main.py --batch [--output_dir <dir>]

Usage (Batch - your list of checkpoints):
    python main.py --batch --checkpoint_list paths.txt --segment_length 4s --output_dir results/
    # paths.txt: one checkpoint path per line (# comments and blank lines ignored)
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Add parent directory and CNN directory to path for imports
# Parent directory allows: from CNN.trainer import ...
# CNN directory allows relative imports within CNN module (from models import ...)
parent_dir = Path(__file__).parent.parent
cnn_dir = parent_dir / 'CNN'
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(cnn_dir) not in sys.path:
    sys.path.insert(0, str(cnn_dir))

from analyzer import SaliencyAnalyzer
from visualization import plot_method_comparison, save_method_comparison_report
from CNN.models import ModelFactory
from CNN.gpu_utils import detect_available_gpus, setup_model_for_gpus, print_gpu_info, get_device
from data_processing.eeg_dataset import EEGDataset, EEGDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute saliency maps for EEG channel importance analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process all best models automatically')
    
    # Required arguments (for single model mode)
    parser.add_argument('--checkpoint', type=str, default=os.environ.get('CHECKPOINT'),
                       help='Path to model checkpoint (best model .pth). Can also set CHECKPOINT env var.')
    parser.add_argument('--target_type', type=str, default=None,
                       choices=['gender', 'age', 'combined'],
                       help='Type of prediction target (required for single model mode)')
    parser.add_argument('--segment_length', type=str, default=None,
                       choices=['1s', '2s', '4s'],
                       help='EEG segment length (required for single model mode)')
    
    # Data arguments (use HDF5_DIR or EEG_HDF5_DIR env var for training-data path, e.g. ~/scratch/processed_eeg_data_hdf5)
    parser.add_argument('--hdf5_dir', type=str,
                       default=os.environ.get('HDF5_DIR') or os.environ.get('EEG_HDF5_DIR') or 'data_processing/processed_eeg_data_hdf5_no_compression',
                       help='Directory containing HDF5 files (multi-part or single). Set HDF5_DIR env to match training data.')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to analyze')
    parser.add_argument('--task_type', type=str, default='both',
                       choices=['active', 'passive', 'both'],
                       help='Task type to analyze')
    
    # Model arguments (CNN and ResNet; auto-detected from checkpoint if not specified)
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type (e.g. gender_cnn, age_cnn, age_resnet34, gender_resnet50). '
                            'Auto-detected from checkpoint if not specified.')
    parser.add_argument('--num_channels', type=int, default=60,
                       help='Number of EEG channels')
    parser.add_argument('--prediction_type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Prediction type (for age models)')
    
    # GPU/Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device preference')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use (1 = single GPU/CPU, >1 = multi-GPU)')
    
    # Saliency method arguments
    # NOTE: Using vanilla_gradients as default for faster testing
    # Integrated gradients is more accurate but ~50x slower (50 steps per sample)
    # Uncomment integrated_gradients when ready for production analysis
    parser.add_argument('--method', type=str, default='vanilla_gradients',  # Changed from 'integrated_gradients' for faster testing
                       choices=['vanilla_gradients', 'integrated_gradients', 'both'],
                       help='Saliency computation method. Use "both" to compute and compare both methods. '
                            'vanilla_gradients is faster (~50x), integrated_gradients is more accurate but slower.')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of steps for integrated gradients')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                       default=os.environ.get('OUTPUT_DIR', 'saliency_results'),
                       help='Directory to save saliency and topography results. Can also set OUTPUT_DIR env var.')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top channels to highlight in visualizations')
    
    # Visualization arguments
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating visualization plots')
    
    # Batch mode arguments
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to process in batch mode (by name). If not specified, processes all models.')
    parser.add_argument('--checkpoint_list', type=str, default=None,
                       help='Text file with one checkpoint path per line (for batch). Overrides BEST_MODELS_CONFIG. '
                            'Requires --segment_length. model_type and target_type are inferred from each checkpoint.')
    
    return parser.parse_args()


# Map checkpoint model_info['model_name'] (class name) to saliency model_type.
# Must match CNN/models/model.py and CNN/resnet/resnet_model.py class names.
_MODEL_NAME_TO_TYPE = {
    # CNN
    'EEGGenderCNN': 'gender_cnn',
    'EEGAgeCNN': 'age_cnn',
    'EEGAgeRegressionCNN': 'age_regression_cnn',
    'CombinedCNN': 'combined_cnn',
    'MultiOutputCNN': 'multi_output_cnn',
    'EEGCNN': 'gender_cnn',  # generic; default to gender_cnn for 2-class
    # ResNet 18
    'EEGGenderResNet': 'gender_resnet',
    'EEGAgeResNet': 'age_resnet',
    # ResNet 34
    'EEGGenderResNet34': 'gender_resnet34',
    'EEGAgeResNet34': 'age_resnet34',
    # ResNet 50
    'EEGGenderResNet50': 'gender_resnet50',
    'EEGAgeResNet50': 'age_resnet50',
    # ResNet other
    'EEGAgeRegressionResNet': 'age_regression_resnet',
    'CombinedResNet': 'combined_resnet',
    'MultiOutputResNet': 'multi_output_resnet',
}


def get_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """
    Infer model type from checkpoint.
    Uses model_info['model_name'] (class name from get_model_info()) and optionally config.
    Only reads metadata; actual weights are loaded later with the correct device in
    analyzer.load_checkpoint -> trainer.load_checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
    
    Returns:
        Model type string (e.g. gender_cnn, age_resnet34, gender_resnet50)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Primary: model_info.model_name is the class name (e.g. EEGGenderResNet34)
    if 'model_info' in checkpoint:
        model_name = checkpoint['model_info'].get('model_name', '')
        if model_name in _MODEL_NAME_TO_TYPE:
            return _MODEL_NAME_TO_TYPE[model_name]
        # Fallback for unknown class names: match by substring (e.g. custom or renamed)
        if model_name:
            if 'ResNet' in model_name or 'resnet' in model_name:
                if 'Gender' in model_name:
                    if 'ResNet34' in model_name:
                        return 'gender_resnet34'
                    if 'ResNet50' in model_name:
                        return 'gender_resnet50'
                    return 'gender_resnet'
                if 'Age' in model_name:
                    if 'Regression' in model_name:
                        return 'age_regression_resnet'
                    if 'ResNet34' in model_name:
                        return 'age_resnet34'
                    if 'ResNet50' in model_name:
                        return 'age_resnet50'
                    return 'age_resnet'
                if 'Combined' in model_name:
                    return 'combined_resnet'
                if 'MultiOutput' in model_name:
                    return 'multi_output_resnet'
            if 'Gender' in model_name or 'gender' in model_name.lower():
                return 'gender_cnn'
            if 'Age' in model_name or 'age' in model_name.lower():
                if 'Regression' in model_name or 'regression' in model_name.lower():
                    return 'age_regression_cnn'
                return 'age_cnn'
            if 'Combined' in model_name or 'combined' in model_name.lower():
                return 'combined_cnn'
            if 'MultiOutput' in model_name or 'multi_output' in model_name.lower():
                return 'multi_output_cnn'
    
    # Fallback: config (TrainingConfig has target_key, prediction_type; no model_type)
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_type = config.get('model_type')
            if model_type and ('resnet' in (model_type or '').lower()):
                return model_type
            target_key = config.get('target_key', '')
            prediction_type = config.get('prediction_type', 'classification')
            if target_key == 'gender':
                return 'gender_cnn'
            if target_key == 'age':
                if prediction_type == 'regression':
                    return 'age_regression_cnn'
                return 'age_cnn'
            if target_key == 'combined':
                return 'combined_cnn'
            if target_key == 'multi_output':
                return 'multi_output_cnn'
    
    # Checkpoint has no model_info/config (e.g. LaBraM: saves 'model' only, no model_info)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model')
    if state_dict is not None:
        keys = list(state_dict.keys())
        # Strip 'module.' prefix for inspection
        key_str = ' '.join(k.replace('module.', '') for k in keys[:30])
        # LaBraM / ViT-style: patch_embed, blocks., encoder., cls_head, head, norm
        labram_indicators = ('patch_embed', 'blocks.', 'encoder.', 'cls_head', 'norm.', 'head.')
        if any(ind in key_str for ind in labram_indicators):
            return 'labram'
    
    raise ValueError(
        f"Could not determine model type from checkpoint: {checkpoint_path}. "
        f"Checkpoint has no 'model_info' or 'config' (expected for CNN/ResNet checkpoints from the CNN trainer). "
        f"Keys in checkpoint: {list(checkpoint.keys())}. "
        f"If this is a LaBraM checkpoint, it is not supported; use a CNN or ResNet .pth instead."
    )


class LaBraMSaliencyWrapper(torch.nn.Module):
    """
    Wraps a LaBraM model so it accepts (B, num_channels, timepoints) input like CNN/ResNet
    and returns logits. Used for saliency: gradients flow to the raw input (B, 60, T).
    """
    PATCH_T = 200

    def __init__(self, labram_model, input_chans):
        super().__init__()
        self.model = labram_model
        self.input_chans = input_chans

    def forward(self, x):
        # x: (B, N, T) e.g. (B, 60, 800) for 4s
        x = x.float() / 100.0
        from einops import rearrange
        # (B, N, T) -> (B, N, A, 200) with A = T // 200
        x = rearrange(x, 'B N (A T) -> B N A T', T=self.PATCH_T)
        out = self.model(x, input_chans=self.input_chans)
        if isinstance(out, dict):
            out = out.get('gender', out.get('age', list(out.values())[0]))
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out


def _create_labram_model(num_classes: int) -> tuple:
    """Create LaBraM model and wrapper. Requires LaBraM on path and timm/einops."""
    labram_dir = parent_dir / 'LaBraM'
    if str(labram_dir) not in sys.path:
        sys.path.insert(0, str(labram_dir))
    import LaBraM.modeling_finetune  # noqa: F401  # register labram_base_patch200_200
    from timm.models import create_model as timm_create_model
    from LaBraM.utils import get_input_chans
    try:
        from utils.eeg_constants import STANDARD_CHANNEL_NAMES
    except ImportError:
        STANDARD_CHANNEL_NAMES = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10',
            'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2',
            'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4',
            'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
        ]
    # Match run_class_finetuning: rel_pos_bias=True by default; abs_pos_emb from args (checkpoint often has pos_embed)
    model = timm_create_model(
        'labram_base_patch200_200',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,  # checkpoint usually has pos_embed; must match training config
        init_values=0.1,
        qkv_bias=True,
        multi_output=False,
    )
    input_chans = get_input_chans(STANDARD_CHANNEL_NAMES)
    wrapper = LaBraMSaliencyWrapper(model, input_chans)
    return wrapper, num_classes


def create_model(model_type: str, num_channels: int, num_classes: int = None,
                 prediction_type: str = 'classification') -> tuple:
    """
    Create model instance (CNN, ResNet, or LaBraM).
    
    Args:
        model_type: Model type string (e.g. gender_cnn, age_resnet34, labram)
        num_channels: Number of channels
        num_classes: Number of classes (auto-determined if None)
        prediction_type: Prediction type
    
    Returns:
        Tuple of (model, num_classes)
    """
    if num_classes is None:
        # CNN
        if model_type == 'gender_cnn':
            num_classes = 2
        elif model_type == 'age_cnn':
            num_classes = 3
        elif model_type == 'age_regression_cnn':
            num_classes = 1
        elif model_type == 'combined_cnn':
            num_classes = 6
        elif model_type == 'multi_output_cnn':
            num_classes = 2  # Will be overridden by separate heads
        # ResNet
        elif model_type in ('gender_resnet', 'gender_resnet34', 'gender_resnet50'):
            num_classes = 2
        elif model_type in ('age_resnet', 'age_resnet34', 'age_resnet50'):
            num_classes = 3
        elif model_type == 'age_regression_resnet':
            num_classes = 1
        elif model_type == 'combined_resnet':
            num_classes = 6
        elif model_type == 'multi_output_resnet':
            num_classes = 2
        # LaBraM: gender binary (nb_classes=1 for BCE), age 3-class
        elif model_type == 'labram':
            num_classes = num_classes if num_classes is not None else 1
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # LaBraM
    if model_type == 'labram':
        return _create_labram_model(num_classes)
    
    # CNN / ResNet
    model = ModelFactory.create_model(
        model_type,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_rate=0.5,
        use_layer_norm=True
    )
    return model, num_classes


def get_model_display_name(model_type: str) -> str:
    """Return short display name for plot titles (e.g. CNN, ResNet34, ResNet50)."""
    if not model_type:
        return 'Model'
    if 'resnet50' in model_type.lower():
        return 'ResNet50'
    if 'resnet34' in model_type.lower():
        return 'ResNet34'
    if 'resnet' in model_type.lower():
        return 'ResNet18'
    if 'labram' in model_type.lower():
        return 'LaBraM'
    return 'CNN'


# Configuration for all best models (for batch mode).
# Update RESNET_BASE and CNN_BASE to your actual directories; batch skips missing files.
# LaBraM: different checkpoint/data format — not supported in this pipeline yet.
RESNET_BASE = 'final_logs/RESNET'  # or e.g. experiment_results, CNN/resnet_results
CNN_BASE = 'CNN/report_2025-12-16'  # your CNN results base path

def _resnet_path(arch: str, task: str, seg: str) -> str:
    """Build ResNet checkpoint path. arch in (18, 34, 50), task in (gender, age), seg in (1s, 2s, 4s)."""
    exp = f"RESNET{arch}_{task}_{seg}"
    name = f"{task}_resnet{arch}_{seg}"  # e.g. gender_resnet18_1s, age_resnet34_4s
    return f"{RESNET_BASE}/{exp}/checkpoints/{name}_best.pth"


BEST_MODELS_CONFIG = [
    # ----- CNN -----
    {'checkpoint': f'{CNN_BASE}/results_1s/checkpoints/gender_baseline_1s_best.pth', 'model_type': 'gender_cnn', 'target_type': 'gender', 'segment_length': '1s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': f'{CNN_BASE}/results_1s/checkpoints/age_classification_1s_best.pth', 'model_type': 'age_cnn', 'target_type': 'age', 'segment_length': '1s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': f'{CNN_BASE}/results_2s/checkpoints/gender_baseline_2s_best.pth', 'model_type': 'gender_cnn', 'target_type': 'gender', 'segment_length': '2s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': f'{CNN_BASE}/results_2s/checkpoints/age_classification_2s_best.pth', 'model_type': 'age_cnn', 'target_type': 'age', 'segment_length': '2s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': f'{CNN_BASE}/results_2s/checkpoints/age_regression_2s_best.pth', 'model_type': 'age_regression_cnn', 'target_type': 'age', 'segment_length': '2s', 'num_classes': 1, 'prediction_type': 'regression', 'class_names': None},
    {'checkpoint': f'{CNN_BASE}/results_4s/checkpoints/gender_baseline_4s_best.pth', 'model_type': 'gender_cnn', 'target_type': 'gender', 'segment_length': '4s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': f'{CNN_BASE}/results_4s/checkpoints/age_classification_4s_best.pth', 'model_type': 'age_cnn', 'target_type': 'age', 'segment_length': '4s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': f'{CNN_BASE}/results_4s/checkpoints/age_regression_4s_best.pth', 'model_type': 'age_regression_cnn', 'target_type': 'age', 'segment_length': '4s', 'num_classes': 1, 'prediction_type': 'regression', 'class_names': None},
    # ----- ResNet 18: gender 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('18', 'gender', '1s'), 'model_type': 'gender_resnet', 'target_type': 'gender', 'segment_length': '1s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('18', 'gender', '2s'), 'model_type': 'gender_resnet', 'target_type': 'gender', 'segment_length': '2s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('18', 'gender', '4s'), 'model_type': 'gender_resnet', 'target_type': 'gender', 'segment_length': '4s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    # ----- ResNet 18: age 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('18', 'age', '1s'), 'model_type': 'age_resnet', 'target_type': 'age', 'segment_length': '1s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('18', 'age', '2s'), 'model_type': 'age_resnet', 'target_type': 'age', 'segment_length': '2s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('18', 'age', '4s'), 'model_type': 'age_resnet', 'target_type': 'age', 'segment_length': '4s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    # ----- ResNet 34: gender 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('34', 'gender', '1s'), 'model_type': 'gender_resnet34', 'target_type': 'gender', 'segment_length': '1s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('34', 'gender', '2s'), 'model_type': 'gender_resnet34', 'target_type': 'gender', 'segment_length': '2s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('34', 'gender', '4s'), 'model_type': 'gender_resnet34', 'target_type': 'gender', 'segment_length': '4s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    # ----- ResNet 34: age 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('34', 'age', '1s'), 'model_type': 'age_resnet34', 'target_type': 'age', 'segment_length': '1s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('34', 'age', '2s'), 'model_type': 'age_resnet34', 'target_type': 'age', 'segment_length': '2s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('34', 'age', '4s'), 'model_type': 'age_resnet34', 'target_type': 'age', 'segment_length': '4s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    # ----- ResNet 50: gender 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('50', 'gender', '1s'), 'model_type': 'gender_resnet50', 'target_type': 'gender', 'segment_length': '1s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('50', 'gender', '2s'), 'model_type': 'gender_resnet50', 'target_type': 'gender', 'segment_length': '2s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    {'checkpoint': _resnet_path('50', 'gender', '4s'), 'model_type': 'gender_resnet50', 'target_type': 'gender', 'segment_length': '4s', 'num_classes': 2, 'prediction_type': 'classification', 'class_names': ['Female', 'Male']},
    # ----- ResNet 50: age 1s, 2s, 4s -----
    {'checkpoint': _resnet_path('50', 'age', '1s'), 'model_type': 'age_resnet50', 'target_type': 'age', 'segment_length': '1s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('50', 'age', '2s'), 'model_type': 'age_resnet50', 'target_type': 'age', 'segment_length': '2s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
    {'checkpoint': _resnet_path('50', 'age', '4s'), 'model_type': 'age_resnet50', 'target_type': 'age', 'segment_length': '4s', 'num_classes': 3, 'prediction_type': 'classification', 'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']},
]


def get_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Extract model name from checkpoint path."""
    return Path(checkpoint_path).stem.replace('_best', '')


def get_target_type_from_model_type(model_type: str) -> str:
    """Infer target_type from model_type (e.g. gender_resnet34 -> gender)."""
    if not model_type:
        return 'gender'
    if model_type.startswith('gender_'):
        return 'gender'
    if model_type.startswith('age_'):
        return 'age'
    if model_type.startswith('combined_'):
        return 'combined'
    if model_type.startswith('multi_output_'):
        return 'age'  # multi-output has both; use age as placeholder
    return 'gender'


def get_prediction_type_from_model_type(model_type: str) -> str:
    """Infer prediction_type from model_type."""
    if 'regression' in model_type.lower():
        return 'regression'
    return 'classification'


def get_class_names_for_target(target_type: str, prediction_type: str):
    """Return class_names for visualization, or None for regression."""
    if prediction_type == 'regression':
        return None
    if target_type == 'gender':
        return ['Female', 'Male']
    if target_type == 'age':
        return ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    if target_type == 'combined':
        return [
            '(Female, <8.5)', '(Female, 8.5-12.5)', '(Female, >12.5)',
            '(Male, <8.5)', '(Male, 8.5-12.5)', '(Male, >12.5)'
        ]
    return None


def load_checkpoint_list(filepath: str, segment_length: str):
    """
    Load a list of checkpoint paths from a file (one path per line; # and blank lines ignored).
    For each path, infer model_type and target_type from the checkpoint; use segment_length for all.
    Returns list of config dicts compatible with BEST_MODELS_CONFIG.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint list file not found: {filepath}")
    lines = path.read_text().strip().splitlines()
    configs = []
    for line in lines:
        line = line.split('#')[0].strip()
        if not line:
            continue
        checkpoint_path = line.strip()
        if not Path(checkpoint_path).exists():
            print(f"⚠️  Checkpoint not found, skipping: {checkpoint_path}")
            continue
        model_type = get_model_type_from_checkpoint(checkpoint_path)
        target_type = get_target_type_from_model_type(model_type)
        prediction_type = get_prediction_type_from_model_type(model_type)
        class_names = get_class_names_for_target(target_type, prediction_type)
        configs.append({
            'checkpoint': checkpoint_path,
            'model_type': model_type,
            'target_type': target_type,
            'segment_length': segment_length,
            'num_classes': 2 if target_type == 'gender' else (3 if target_type == 'age' and prediction_type == 'classification' else 1),
            'prediction_type': prediction_type,
            'class_names': class_names,
        })
    return configs


def process_single_model(args):
    """Process a single model (original functionality)."""
    print("=" * 80)
    print("EEG Channel Importance Analysis via Saliency Maps")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target type: {args.target_type}")
    print(f"Segment length: {args.segment_length}")
    print(f"Method: {args.method}")
    print("=" * 80)
    
    # Determine model type
    if args.model_type:
        model_type = args.model_type
    else:
        print("Auto-detecting model type from checkpoint...")
        model_type = get_model_type_from_checkpoint(args.checkpoint)
        print(f"Detected model type: {model_type}")
    
    # Create model (for LaBraM, num_classes: 1=gender, 3=age)
    print(f"\nCreating model: {model_type}")
    if model_type == 'labram':
        labram_num_classes = 1 if args.target_type == 'gender' else 3
        model, num_classes = create_model(model_type, args.num_channels,
                                          num_classes=labram_num_classes,
                                          prediction_type=args.prediction_type)
    else:
        model, num_classes = create_model(model_type, args.num_channels,
                                          prediction_type=args.prediction_type)
    
    # Setup device and multi-GPU if needed
    device = get_device(args.device)
    available_gpus = detect_available_gpus()
    
    if args.num_gpus > 1 and available_gpus > 0:
        print(f"\nSetting up multi-GPU ({args.num_gpus} GPUs)...")
        print(f"  Using PyTorch DataParallel for multi-GPU acceleration")
        print(f"  Note: Saliency computation processes samples individually, but model forward/backward")
        print(f"        passes will be distributed across {args.num_gpus} GPUs for faster computation.")
        model = setup_model_for_gpus(model, args.num_gpus, device)
        print_gpu_info(args.num_gpus, args.num_gpus, device)
        print(f"  ✅ Multi-GPU setup complete. Model will use {args.num_gpus} GPUs for computation.")
    else:
        if args.num_gpus > 1:
            print(f"\n⚠️  Requested {args.num_gpus} GPUs but only {available_gpus} available. Using single GPU/CPU.")
        model = model.to(device)
        if device.type == 'cuda':
            print(f"  ✅ Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ✅ Using CPU")
    
    # Display name for plot titles (e.g. CNN, ResNet34, LaBraM)
    model_display_name = get_model_display_name(model_type)
    # Create analyzer (segment_length and model_display_name used in plot titles)
    analyzer = SaliencyAnalyzer(
        model=model,
        target_type=args.target_type,
        num_channels=args.num_channels,
        device=device,
        device_preference=args.device,
        prediction_type=args.prediction_type,
        segment_length=args.segment_length,
        model_display_name=model_display_name
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    if model_type == 'labram':
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model' not in ckpt:
            raise KeyError(f"LaBraM checkpoint must contain 'model' key. Keys: {list(ckpt.keys())}")
        labram_model = model.module.model if isinstance(model, torch.nn.DataParallel) else model.model
        labram_model.load_state_dict(ckpt['model'], strict=True)
        print(f"✅ Loaded LaBraM checkpoint from {args.checkpoint}")
    else:
        analyzer.load_checkpoint(args.checkpoint)
    
    # Create data loader (expand ~ to match training data path, e.g. ~/scratch/processed_eeg_data_hdf5)
    hdf5_dir_raw = os.path.expanduser(args.hdf5_dir)
    print(f"\nLoading data from: {hdf5_dir_raw}")
    hdf5_dir = Path(hdf5_dir_raw)
    
    # Get HDF5 file path(s) - handles both single files (2s/4s) and multiple files (1s)
    # Note: For 1s segments, this returns lists of files. For 2s/4s, returns single files.
    train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
        hdf5_dir, args.segment_length
    )
    
    # Select the appropriate split
    if args.split == 'train':
        split_files = train_files
    elif args.split == 'val':
        split_files = val_files
    else:  # test
        split_files = test_files
    
    # Verify only the split we use exists (saliency does not require train/val)
    def _check_split_files(files, split_name):
        if isinstance(files, (list, tuple)):
            if len(files) == 0:
                raise FileNotFoundError(
                    f"{split_name} HDF5 files not found in {hdf5_dir}. "
                    f"Use the same directory as training (e.g. --hdf5_dir ~/scratch/processed_eeg_data_hdf5 or set HDF5_DIR)."
                )
            for p in files:
                if not Path(p).exists():
                    raise FileNotFoundError(f"{split_name} HDF5 file not found: {p}")
        else:
            if not Path(files).exists():
                raise FileNotFoundError(
                    f"{split_name} HDF5 file not found: {files}. "
                    f"Use --hdf5_dir to point to your training data directory (e.g. ~/scratch/processed_eeg_data_hdf5)."
                )
    _check_split_files(split_files, args.split)
    
    # Create dataset - handles both single files and lists of files automatically
    dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=split_files,
        task_type=args.task_type,
        target_type=args.target_type,
        transform=None,
        gender_transform=None,
        age_transform=None,
        combined_transform=None,
        user_identification_transform=None,
        shuffle=False  # Don't shuffle for consistent analysis
    )
    
    # For saliency computation, use fewer workers to avoid hanging issues
    # Saliency computation is CPU-bound per sample, so too many workers can cause issues
    # Also, IterableDataset with many workers can sometimes hang
    num_workers = 0 if device.type == 'cuda' else 0  # Use 0 workers to avoid multiprocessing issues
    print(f"\nCreating data loader (batch_size={args.batch_size}, num_workers={num_workers})...")
    
    data_loader = EEGDataLoader.create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,  # Use 0 to avoid hanging with IterableDataset
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"✅ Data loader created successfully")
    
    # Determine which methods to run
    if args.method == 'both':
        methods_to_run = ['vanilla_gradients', 'integrated_gradients']
    else:
        methods_to_run = [args.method]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine class names based on target type
    # Use stored class_names from batch mode if available, otherwise determine from target_type
    if hasattr(args, '_class_names') and args._class_names is not None:
        class_names = args._class_names
    elif args.target_type == 'gender':
        class_names = ['Female', 'Male']
    elif args.target_type == 'age':
        class_names = ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    elif args.target_type == 'combined':
        class_names = [
            '(Female, <8.5)', '(Female, 8.5-12.5)', '(Female, >12.5)',
            '(Male, <8.5)', '(Male, 8.5-12.5)', '(Male, >12.5)'
        ]
    else:
        class_names = None
    
    # Process each method
    all_analyses = {}
    for method in methods_to_run:
        print(f"\n{'='*80}")
        print(f"Processing method: {method.upper()}")
        print(f"{'='*80}")
        
        # Compute saliency
        print(f"\nComputing saliency maps...")
        method_kwargs = {}
        if method == 'integrated_gradients':
            method_kwargs['num_steps'] = args.num_steps
        
        results = analyzer.compute_saliency(
            data_loader=data_loader,
            method=method,
            max_samples=args.max_samples,
            **method_kwargs
        )
        
        # Analyze channel importance
        # Note: This analyzes channel importance across ALL participants (aggregated)
        # The current implementation generates one saliency map aggregated across all participants
        # by computing the mean channel importance across all samples.
        print(f"\nAnalyzing channel importance (aggregated across all participants)...")
        analysis = analyzer.analyze_channel_importance(method=method)
        all_analyses[method] = analysis
        
        # Generate aggregated saliency map across all participants
        # This provides a single representative saliency map for the entire population
        print(f"\nGenerating aggregated saliency map across all participants...")
        aggregated = analyzer.aggregate_across_participants(method=method, aggregation='mean')
        print(f"  ✅ Aggregated map represents {aggregated['num_participants']} participants "
              f"with {aggregated['num_samples']} total samples")
        
        # Save results
        print(f"\nSaving results to: {output_dir}")
        analyzer.save_results(str(output_dir), method=method)
        
        # Generate visualizations
        if not args.no_plots:
            print(f"\nGenerating visualizations for {method}...")
            analyzer.visualize_results(
                output_dir=str(output_dir),
                method=method,
                channel_names=None,  # Can be extended with actual channel names
                class_names=class_names,
                top_k=args.top_k
            )
        
        # Print summary for this method
        print(f"\n{'-'*80}")
        print(f"Summary for {method.upper()}:")
        print(f"{'-'*80}")
        print(f"Top {args.top_k} most important channels:")
        top_channels = analyzer.get_top_channels(method=method, top_k=args.top_k)
        for rank, ch_idx in enumerate(top_channels, 1):
            avg_imp = analysis['average_importance'][ch_idx]
            print(f"  {rank:2d}. Channel {ch_idx+1:2d} (importance: {avg_imp:.6f})")
    
    # If both methods were run, generate comparison
    if args.method == 'both' and len(methods_to_run) == 2:
        print(f"\n{'='*80}")
        print("Generating Method Comparison")
        print(f"{'='*80}")
        
        # Get results for both methods
        vg_results = analyzer.results.get('vanilla_gradients')
        ig_results = analyzer.results.get('integrated_gradients')
        
        if vg_results is not None and ig_results is not None:
            # Generate comparison visualization
            if not args.no_plots:
                print("\nGenerating comparison visualizations...")
                title_suffix = ", ".join(filter(None, [args.segment_length, model_display_name]))
                plot_method_comparison(
                    channel_importance_vg=vg_results['channel_importance'],
                    channel_importance_ig=ig_results['channel_importance'],
                    channel_names=None,
                    title=f"Saliency Method Comparison - {args.target_type.capitalize()} ({title_suffix})",
                    save_path=str(output_dir / "method_comparison.png"),
                    top_k=args.top_k
                )
            
            # Generate comparison report
            print("\nGenerating comparison report...")
            save_method_comparison_report(
                channel_importance_vg=vg_results['channel_importance'],
                channel_importance_ig=ig_results['channel_importance'],
                channel_names=None,
                save_path=str(output_dir / "method_comparison_report.csv")
            )
        else:
            print("⚠️  Warning: Could not generate comparison - missing results for one or both methods")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    
    if args.method == 'both':
        print(f"\nMethods processed: {', '.join(methods_to_run)}")
        print(f"Comparison report: {output_dir / 'method_comparison_report.csv'}")
        if not args.no_plots:
            print(f"Comparison visualization: {output_dir / 'method_comparison.png'}")
    
    print("=" * 80)


def process_batch_models(args):
    """Process all best models in batch mode."""
    print("=" * 80)
    print("SALIENCY MAP COMPUTATION FOR ALL BEST MODELS")
    print("=" * 80)
    print(f"HDF5 directory: {args.hdf5_dir}")
    print(f"Device preference: {args.device}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Method: {args.method}")
    print(f"Max samples per model: {args.max_samples if args.max_samples else 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Check GPU availability
    available_gpus = detect_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")
    if args.num_gpus > 1 and available_gpus == 0:
        print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but CUDA is not available.")
        print("   Will use CPU instead.")
        args.num_gpus = 1
        args.device = 'cpu'
    elif args.num_gpus > available_gpus:
        print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available.")
        print(f"   Will use {available_gpus} GPUs instead.")
        args.num_gpus = available_gpus
    
    # Use checkpoint list file if provided; otherwise BEST_MODELS_CONFIG
    if args.checkpoint_list:
        if not args.segment_length:
            raise ValueError(
                "When using --checkpoint_list you must specify --segment_length (e.g. --segment_length 4s), "
                "since segment length is not stored in the checkpoint."
            )
        print(f"Loading checkpoint list: {args.checkpoint_list} (segment_length={args.segment_length})")
        models_to_process = load_checkpoint_list(args.checkpoint_list, args.segment_length)
        if not models_to_process:
            raise ValueError(f"No valid checkpoint paths found in {args.checkpoint_list}")
        print(f"  Found {len(models_to_process)} checkpoint(s)")
    else:
        models_to_process = BEST_MODELS_CONFIG
        if args.models:
            models_to_process = [
                config for config in BEST_MODELS_CONFIG
                if get_model_name_from_checkpoint(config['checkpoint']) in args.models
            ]
            if not models_to_process:
                print(f"\n❌ No matching models found for: {args.models}")
                print(f"   Available models: {[get_model_name_from_checkpoint(c['checkpoint']) for c in BEST_MODELS_CONFIG]}")
                return
    
    print(f"\nProcessing {len(models_to_process)} model(s)...")
    print("=" * 80)
    
    # Import tqdm for progress tracking
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs):
            return iterable
    
    # Process each model
    successful = 0
    failed = 0
    
    # Create progress bar for models
    if TQDM_AVAILABLE:
        model_pbar = tqdm(
            enumerate(models_to_process, 1),
            total=len(models_to_process),
            desc="Processing models",
            unit="model",
            dynamic_ncols=True
        )
    else:
        model_pbar = enumerate(models_to_process, 1)
    
    for i, config in model_pbar:
        model_name = get_model_name_from_checkpoint(config['checkpoint'])
        
        # Update progress bar
        if TQDM_AVAILABLE:
            model_pbar.set_description(f"Processing: {model_name}")
            model_pbar.set_postfix({
                'successful': successful,
                'failed': failed
            })
        
        print(f"\n[{i}/{len(models_to_process)}] Processing: {model_name}")
        print("-" * 80)
        
        if not Path(config['checkpoint']).exists():
            print(f"⚠️  Checkpoint not found: {config['checkpoint']}")
            print("   Skipping this model...")
            failed += 1
            continue
        
        try:
            # Create temporary args for this model
            import argparse
            model_args = argparse.Namespace(**vars(args))
            model_args.checkpoint = config['checkpoint']
            model_args.target_type = config['target_type']
            model_args.segment_length = config['segment_length']
            model_args.model_type = config['model_type']
            model_args.prediction_type = config['prediction_type']
            model_args.output_dir = str(Path(args.output_dir) / model_name)
            
            # Store class_names for visualization (will be used in process_single_model)
            model_args._class_names = config.get('class_names')
            
            # Process this model
            process_single_model(model_args)
            successful += 1
            
            # Update progress
            if TQDM_AVAILABLE:
                model_pbar.set_postfix({
                    'successful': successful,
                    'failed': failed,
                    'status': '✓'
                })
            
            # Clear CUDA cache after each model to prevent state corruption
            if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
            # Update progress
            if TQDM_AVAILABLE:
                model_pbar.set_postfix({
                    'successful': successful,
                    'failed': failed,
                    'status': '✗'
                })
            
            # Clear CUDA cache on error to reset state
            if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Close progress bar
    if TQDM_AVAILABLE:
        model_pbar.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total models processed: {len(models_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


def main():
    """Main function."""
    args = parse_args()
    
    # Validate arguments based on mode
    if args.batch:
        # Batch mode: process all best models
        process_batch_models(args)
    else:
        # Single model mode: validate required arguments
        if not args.checkpoint or not args.target_type or not args.segment_length:
            print("❌ Error: Single model mode requires --checkpoint, --target_type, and --segment_length")
            print("   Or use --batch to process all best models automatically")
            return
        process_single_model(args)


if __name__ == '__main__':
    main()


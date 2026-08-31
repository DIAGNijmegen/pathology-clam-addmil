from __future__ import print_function

# Refactor of create_heatmaps.py:
#   - `if __name__ == '__main__':` became a proper main(), decomposed into named
#     functions grouped by concern: data/model loading (build_arg_namespaces,
#     load_default_params, build_process_stack, load_model_and_encoder),
#     segmentation (segment_and_visualize, resolve_per_slide_params), inference
#     (run_coarse_inference, sample_topk_patches), and plotting (draw_blockmap,
#     draw_final_heatmap, save_original_slide), orchestrated per slide by
#     process_slide().
#   - data_arguments.provided_mask_dir (optional): if a mask already exists there
#     for a slide (e.g. produced upstream by slide2vec's own tissue detection),
#     segmentation is skipped and that mask is reused -- see segment_and_visualize
#     / heatmap_utils_refractored.initialize_wsi.
#   - data_arguments.coord_dir (optional): if a `<slide_id>.npy` coordinates file
#     already exists there (e.g. slide2vec's own post-segmentation tile
#     coordinates), both segmentation AND contour-based patch generation are
#     skipped -- patches are read directly from those coordinates via
#     Wsi_Region_FromCoords. When the file has a `tile_size_lv0` field, it
#     overrides the configured patch_size. The tile footprints are also traced
#     into a standard CLAM `<slide_id>_mask.pkl` (WholeSlideImage.
#     segmentTissueFromCoords), so mask visualization and tissue-aware heatmap
#     masking work as they do for a normally-segmented slide. See
#     build_wsi_from_provided_coords / load_provided_coords.
#   - heatmap_arguments.scaling (optional, default 'percentile'): choose 'percentile'
#     or 'minmax' normalization for the plotted heatmap. Forced to 'none' when
#     use_ref_scores is on, since those scores are already normalized against a
#     reference distribution and shouldn't be rescaled again.
#   - infer_single_slide() is gone; the former two-step inference (a features-only
#     pass, a separate whole-bag prediction pass, and a second, redundant
#     per-batch attention pass) is now one call to compute_from_patches(), which
#     extracts every patch's features once and runs the model once on the
#     assembled bag (see heatmap_utils_refractored.py for the shared logic).
# Feature-extraction caching (skip re-running the encoder if a slide's .pt/.h5
# already exist) and the original ROI-heatmap fallback/skip behavior are preserved.

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.eval_utils import initiate_model as initiate_model
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from models import get_encoder
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from wsi_core.WholeSlideImage import WholeSlideImage
from vis_utils.heatmap_utils_refractored import initialize_wsi, drawHeatmap, compute_from_patches, infer_slide, save_hdf5
from wsi_core.wsi_utils import sample_rois
from dataset_modules.wsi_dataset import Wsi_Region_FromCoords
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()
    return params


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_arg_namespaces(config_dict):
    patch_args = argparse.Namespace(**config_dict['patching_arguments'])
    data_args = argparse.Namespace(**config_dict['data_arguments'])
    model_args = config_dict['model_arguments']
    model_args.update({'n_classes': config_dict['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = argparse.Namespace(**config_dict['encoder_arguments'])
    exp_args = argparse.Namespace(**config_dict['exp_arguments'])
    heatmap_args = argparse.Namespace(**config_dict['heatmap_arguments'])
    sample_args = argparse.Namespace(**config_dict['sample_arguments'])
    return patch_args, data_args, model_args, encoder_args, exp_args, heatmap_args, sample_args


def load_default_params(data_args):
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
                       'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
    preset = data_args.preset
    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]
        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]
        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]
        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]
    return def_seg_params, def_filter_params, def_vis_params, def_patch_params


def build_process_stack(data_args, def_seg_params, def_filter_params, def_vis_params, def_patch_params):
    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                            use_heatmap_args=False)
    else:
        df = pd.read_csv(data_args.process_list)
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                            use_heatmap_args=False)
    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))
    return process_stack

def build_img_transforms(encoder_args):
    """The encoder's eval-time image transforms, built from MODEL2CONSTANTS
    without constructing (or downloading) the encoder itself. Enough for slides
    whose patch features are precomputed but whose coordinate dataset still needs
    a transform to instantiate."""
    constants = MODEL2CONSTANTS[encoder_args.model_name]
    return get_eval_transforms(mean=constants['mean'], std=constants['std'],
                               target_img_size=encoder_args.target_img_size)

def encoder_is_needed(process_stack, data_args):
    """True unless every slide to be processed already has a `<slide_id>.pt` in
    data_args.features_dir -- in which case CLAM's own feature extractor is never
    run and does not need to be loaded."""
    features_dir = getattr(data_args, 'features_dir', None)
    if not features_dir:
        return True
    for slide_id in process_stack['slide_id'].astype(str):
        if data_args.slide_ext in slide_id:
            slide_id = slide_id.replace(data_args.slide_ext, '')
        if not os.path.isfile(os.path.join(features_dir, slide_id + '.pt')):
            return True
    return False


def load_model_and_encoder(model_args, encoder_args, load_encoder=True):
    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError
    if load_encoder:
        feature_extractor, img_transforms = get_encoder(encoder_args.model_name,
                                                         target_img_size=encoder_args.target_img_size)
        _ = feature_extractor.eval()
        feature_extractor = feature_extractor.to(device)
    else:
        print('all slides have precomputed features -- skipping feature extractor load')
        feature_extractor = None
        img_transforms = build_img_transforms(encoder_args)
    print('Done!')
    return model, feature_extractor, img_transforms


def build_reverse_label_dict(data_args):
    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    return {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}


# ---------------------------------------------------------------------------
# Per-slide bookkeeping
# ---------------------------------------------------------------------------

def resolve_slide_identity(process_stack, i, data_args, reverse_label_dict):
    slide_name = process_stack.loc[i, 'slide_id']
    if data_args.slide_ext not in slide_name:
        slide_name += data_args.slide_ext
    try:
        label = process_stack.loc[i, 'label']
    except KeyError:
        label = 'Unspecified'
    slide_id = slide_name.replace(data_args.slide_ext, '')
    grouping = reverse_label_dict[label] if not isinstance(label, str) else label
    return slide_name, slide_id, label, grouping


def resolve_slide_path(data_args, process_stack, i, slide_name):
    if isinstance(data_args.data_dir, str):
        return os.path.join(data_args.data_dir, slide_name)
    elif isinstance(data_args.data_dir, dict):
        data_dir_key = process_stack.loc[i, data_args.data_dir_key]
        return os.path.join(data_args.data_dir[data_dir_key], slide_name)
    raise NotImplementedError


def resolve_roi_bounds(heatmap_args, process_stack, i):
    if heatmap_args.use_roi:
        x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
        y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
        return (int(x1), int(y1)), (int(x2), int(y2))
    return None, None


def resolve_per_slide_params(row, def_seg_params=None, def_filter_params=None, def_vis_params=None):
    seg_params = def_seg_params.copy() if def_seg_params is not None else {}
    filter_params = def_filter_params.copy() if def_filter_params is not None else {}
    vis_params = def_vis_params.copy() if def_vis_params is not None else {}
    seg_params = load_params(row, seg_params) if def_seg_params is not None else {}
    filter_params = load_params(row, filter_params) if def_filter_params is not None else {}
    vis_params = load_params(row, vis_params) if def_vis_params is not None else {}
    #
    # If seg_params is not empty
    if seg_params:
        keep_ids = str(seg_params['keep_ids'])
        exclude_ids = str(seg_params['exclude_ids'])
        #
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []
        #
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []
    return seg_params, filter_params, vis_params


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_and_visualize(slide_path, mask_file, mask_path, seg_params, filter_params, vis_params,
                           provided_mask_path=None):
    """
    Build the WSI object (segmenting tissue, unless a provided mask short-circuits
    that -- see initialize_wsi) and save a visualization of the segmentation mask.
    """
    print('Initializing WSI object')
    wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                 filter_params=filter_params, provided_mask_path=provided_mask_path)
    print('Done!')
    visualize_and_save_mask(wsi_object, vis_params, mask_path)
    return wsi_object


def visualize_and_save_mask(wsi_object, vis_params, mask_path):
    if vis_params['vis_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        vis_params['vis_level'] = best_level
    mask = wsi_object.visWSI(**vis_params, number_contours=True)
    mask.save(mask_path)


def load_provided_coords(coord_path, default_patch_size):
    """
    Load pre-computed patch coordinates produced by an upstream pipeline (e.g.
    slide2vec, which runs its own tissue segmentation) together with the level-0
    patch size to read them at.
    Two on-disk layouts are accepted:
      - a structured array with at least `x` and `y` fields (slide2vec style); a
        `tile_size_lv0` field, when present, overrides `default_patch_size`.
      - a plain (N, 2) array of level-0 (x, y) coordinates.
    Returns (coords, patch_size): coords are level-0 top-left corners as an
    (N, 2) int array; patch_size is a (w, h) tuple in level-0 pixels. Reading is
    always done at patch_level 0 -- see build_wsi_from_provided_coords.
    """
    arr = np.load(coord_path, allow_pickle=False)
    if arr.dtype.names is not None:
        if 'x' not in arr.dtype.names or 'y' not in arr.dtype.names:
            raise ValueError("structured coords file {} lacks 'x'/'y' fields".format(coord_path))
        coords = np.stack([arr['x'], arr['y']], axis=1).astype(int)
        if 'tile_size_lv0' in arr.dtype.names and len(arr) > 0:
            tile_size = int(np.median(np.asarray(arr['tile_size_lv0'])))
            patch_size = (tile_size, tile_size)
        else:
            patch_size = tuple(default_patch_size)
    else:
        coords = np.asarray(arr).reshape(-1, 2).astype(int)
        patch_size = tuple(default_patch_size)
    if len(coords) == 0:
        raise ValueError('no coordinates found in {}'.format(coord_path))
    return coords, patch_size


def build_wsi_from_provided_coords(slide_path, coord_path, patch_args, img_transforms, vis_params, mask_path, mask_file):
    """
    Build the WSI object and its patch dataset from a pre-computed coordinates
    file (post-segmentation tiles from an upstream pipeline), skipping CLAM's own
    tissue segmentation and contour-based patch generation.
    The provided tiles are additionally converted into a standard CLAM
    segmentation and written to `mask_file` as `<slide_id>_mask.pkl`: their
    footprints are traced into tissue contours by
    WholeSlideImage.segmentTissueFromCoords, so mask visualization and
    tissue-aware heatmap masking (visHeatmap / get_seg_mask) behave exactly as
    for a normally-segmented slide. An existing mask_file is loaded as-is instead
    of being rebuilt.
    """
    if patch_args.patch_level != 0:
        raise NotImplementedError('provided coordinates are interpreted at level 0; '
                                  'set patching_arguments.patch_level: 0')
    wsi_object = WholeSlideImage(slide_path)
    default_patch_size = tuple(patch_args.patch_size for _ in range(2))
    coords, patch_size = load_provided_coords(coord_path, default_patch_size)
    if os.path.isfile(mask_file):
        print('Using existing segmentation mask: ', mask_file)
        wsi_object.initSegmentation(mask_file)
    else:
        wsi_object.segmentTissueFromCoords(coords, patch_size)
        wsi_object.saveSegmentation(mask_file)
        print('Saved segmentation derived from provided coords: ', mask_file)
    visualize_and_save_mask(wsi_object, vis_params, mask_path)
    roi_dataset = Wsi_Region_FromCoords(wsi_object, coords=coords, patch_size=patch_size, level=patch_args.patch_level, t=img_transforms, custom_downsample=patch_args.custom_downsample)
    return wsi_object, roi_dataset, patch_size



# ---------------------------------------------------------------------------
# Inference (coarse whole-slide pass + patch sampling)
# ---------------------------------------------------------------------------
def load_provided_features(feat_path, expected_dim=None, expected_n=None):
    """
    Load a precomputed patch-embedding file and return it in CLAM's on-disk
    layout: a plain CPU float32 tensor of shape (N, D), one row per patch,
    row-aligned with the slide's patch coordinates.
    Accepts a bare tensor or ndarray, or a dict keyed by one of
    features/feature/embeddings/embedding/x; a leading singleton batch dim is
    squeezed. Raises if the result is not 2-D, or if `expected_dim` /
    `expected_n` are supplied and disagree (feature dim vs the model's embed_dim,
    row count vs the number of coordinates for this slide).
    """
    try:
        obj = torch.load(feat_path, map_location='cpu')
    except Exception:
        # torch>=2.6 defaults weights_only=True, which rejects ndarray/dict
        # containers; the user opted into this file via features_dir, so trust it.
        obj = torch.load(feat_path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for key in ('features', 'feature', 'embeddings', 'embedding', 'x'):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError('{}: dict has no recognized feature key (keys: {})'.format(
                feat_path, list(obj.keys())))
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not torch.is_tensor(obj):
        raise TypeError('{}: unsupported feature container {}'.format(feat_path, type(obj)))
    #
    feats = obj.detach().to(torch.float32).cpu().contiguous()
    if feats.dim() == 3 and feats.size(0) == 1:
        feats = feats.squeeze(0)
    if feats.dim() != 2:
        raise ValueError('{}: expected a 2-D (N, D) feature tensor, got shape {}'.format(
            feat_path, tuple(feats.shape)))
    if expected_dim is not None and feats.size(1) != expected_dim:
        raise ValueError('{}: feature dim {} != model embed_dim {}'.format(
            feat_path, feats.size(1), expected_dim))
    if expected_n is not None and feats.size(0) != expected_n:
        raise ValueError('{}: {} feature rows but {} coordinates for this slide'.format(
            feat_path, feats.size(0), expected_n))
    print('using provided features {}: {} x {}'.format(feat_path, feats.size(0), feats.size(1)))
    return feats


def run_coarse_inference(wsi_object, img_transforms, feature_extractor, model, model_args, exp_args,
                          h5_path, features_path, attn_coord_save_path, logits_coord_save_path, blocky_wsi_kwargs, roi_dataset=None,
                          provided_features_path=None, provided_coords=None):
    """
    One inference step for the whole slide. Patch features come from, in priority
    order:
      1. a user-supplied embedding file (`provided_features_path`): the encoder is
         skipped, the file is converted to CLAM's plain (N, D) float tensor
         (load_provided_features) and the model is run on it directly. The
         converted tensor / coords / attention are then cached exactly like an
         extracted slide, so downstream code and future runs are identical.
      2. a `<slide_id>.pt` cached from a previous run (only the cheap model
         forward is redone, to refresh the prediction / attention cache).
      3. fresh extraction over the patches via compute_from_patches().
    `roi_dataset`, when given (e.g. built from a provided coordinates file), is
    used instead of contour-generated patches. `provided_coords` (N, 2) must
    accompany provided features -- each embedding row is a patch at that
    coordinate; if omitted it is read back from an existing `<slide_id>.h5`.
    """
    if provided_features_path is not None and os.path.isfile(provided_features_path):
        if provided_coords is None:
            if not os.path.isfile(h5_path):
                raise ValueError('provided features {} need matching coordinates: set '
                                 'data_arguments.coordinates_dir for this slide'.format(provided_features_path))
            with h5py.File(h5_path, 'r') as f:
                provided_coords = f['coords'][:]
        coords = np.asarray(provided_coords)
        features = load_provided_features(provided_features_path, expected_dim=getattr(model_args, 'embed_dim', None), expected_n=len(coords))
        Y_hat, ids, probs, A, patch_logits = infer_slide(model, features, model_type=model_args.model_type, k=exp_args.n_classes)
        save_hdf5(h5_path, {'features': features.numpy(), 'coords': coords}, mode='w')
        save_hdf5(attn_coord_save_path, {'attention_scores': A, 'coords': coords}, mode='w')
        save_hdf5(logits_coord_save_path, {'patch_logits': patch_logits, 'coords': coords}, mode='w')
        torch.save(features, features_path)
    elif os.path.isfile(features_path):
        features = torch.load(features_path)
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
        Y_hat, ids, probs, A, patch_logits = infer_slide(model, features, model_type=model_args.model_type, k=exp_args.n_classes)
        save_hdf5(attn_coord_save_path, {'attention_scores': A, 'coords': coords}, mode='w')
        save_hdf5(logits_coord_save_path, {'patch_logits': patch_logits, 'coords': coords}, mode='w')
    else:
        features, coords, Y_hat, ids, probs, A, patch_logits = compute_from_patches(
            wsi_object=wsi_object, img_transforms=img_transforms, feature_extractor=feature_extractor,
            model=model, model_type=model_args.model_type, batch_size=exp_args.batch_size,
            attn_save_path=attn_coord_save_path, logits_coord_save_path=logits_coord_save_path,feat_save_path=h5_path, clam_pred=None,
            k=exp_args.n_classes, roi_dataset=roi_dataset, **blocky_wsi_kwargs)
        torch.save(features, features_path)
    return features, coords, Y_hat, ids, probs, A, patch_logits


def sample_topk_patches(scores, coords, sample_args, wsi_object, label, Y_hat, patch_args, exp_args, slide_id):
    for sample in sample_args.samples:
        if not sample['sample']:
            continue
        tag = "label_{}_pred_{}".format(label, Y_hat)
        sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code,
                                        'sampled_patches', str(tag), sample['name'])
        os.makedirs(sample_save_dir, exist_ok=True)
        print('sampling {}'.format(sample['name']))
        sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                      score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
        for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
            print('coord: {} score: {:.3f}'.format(s_coord, s_score))
            patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level,
                                                (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
            patch.save(os.path.join(sample_save_dir,
                                     '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def draw_blockmap(heatmap_save_path, scores, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, r_slide_save_dir, slide_id):
    if os.path.isfile(heatmap_save_path):
        return
    scaling = getattr(heatmap_args, 'scaling', 'minmax') 
    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                           alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=heatmap_args.vis_level,
                           blank_canvas=False, thresh=-1, patch_size=vis_patch_size, scaling=scaling)
    heatmap.save(heatmap_save_path)
    del heatmap

def draw_final_heatmap(scores, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, patch_args,
                        top_left, bot_right, p_slide_save_dir, slide_id):
    # if ref_scores were already used to normalize `scores` upstream, don't rescale again here
    scaling = 'none' if heatmap_args.use_ref_scores else getattr(heatmap_args, 'scaling', 'minmax')
    heatmap_vis_args = {'scaling': scaling, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur,
                         'custom_downsample': heatmap_args.custom_downsample}
    heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(
        slide_id, float(patch_args.overlap), int(heatmap_args.use_roi), int(heatmap_args.blur),
        int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), float(heatmap_args.alpha),
        int(heatmap_args.vis_level), int(heatmap_args.binarize), float(heatmap_args.binary_thresh),
        heatmap_args.save_ext)
    if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
        return
    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                           alpha=heatmap_args.alpha, **heatmap_vis_args, binarize=heatmap_args.binarize,
                           blank_canvas=heatmap_args.blank_canvas, thresh=heatmap_args.binary_thresh,
                           patch_size=vis_patch_size, overlap=patch_args.overlap, top_left=top_left,
                           bot_right=bot_right)
    if heatmap_args.save_ext == 'jpg':
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
    else:
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))


def save_original_slide(img_path, wsi_object, heatmap_args, vis_level, p_slide_save_dir, slide_id):
    if os.path.isfile(img_path):
        return
    heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
    if heatmap_args.save_ext == 'jpg':
        heatmap.save(img_path, quality=100)
    else:
        heatmap.save(img_path)



# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(i, process_stack, patch_args, data_args, exp_args, heatmap_args, sample_args, model_args, def_seg_params, def_filter_params, def_vis_params, patch_size, step_size, model, feature_extractor, img_transforms, reverse_label_dict):
    slide_name, slide_id, label, grouping = resolve_slide_identity(process_stack, i, data_args, reverse_label_dict)
    slide_path = resolve_slide_path(data_args, process_stack, i, slide_name)
    print('\nprocessing: ', slide_name)
    print('slide id: ', slide_id)
    p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
    r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
    os.makedirs(p_slide_save_dir, exist_ok=True)
    os.makedirs(r_slide_save_dir, exist_ok=True)
    top_left, bot_right = resolve_roi_bounds(heatmap_args, process_stack, i)
    #
    ###################################################################
    # Coordinates handling
    mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')
    mask_path = os.path.join(r_slide_save_dir, slide_id + '_mask.jpg')
    seg_params, filter_params, vis_params = resolve_per_slide_params(process_stack.loc[i], def_seg_params, def_filter_params, def_vis_params)
    # If patch coordinates were already computed upstream (e.g. by slide2vec, which does its own tissue segmentation), use them directly and skip both CLAM's tissue segmentation and its contour-based patch generation.
    coord_dir = getattr(data_args, 'coordinates_dir', None)
    coord_path = os.path.join(coord_dir, slide_id + '.npy') if coord_dir is not None else None
    use_provided_coords = coord_path is not None and os.path.isfile(coord_path)
    if use_provided_coords:
        wsi_object, roi_dataset, slide_patch_size = build_wsi_from_provided_coords(slide_path, coord_path, patch_args, img_transforms, vis_params, mask_path, mask_file)
        vis_patch_size = tuple((np.array(slide_patch_size) * patch_args.custom_downsample).astype(int))
    else:
        # If a mask was already produced upstream instead, reuse it and skip only CLAM's segmentation step (patches are still contour-generated).
        # provided_mask_dir = getattr(data_args, 'provided_mask_dir', None)
        # provided_mask_path = (os.path.join(provided_mask_dir, slide_id + '_mask.pkl') if provided_mask_dir is not None else None)
        wsi_object = segment_and_visualize(slide_path, mask_file, mask_path, seg_params, filter_params, vis_params, provided_mask_path=None)
        roi_dataset = None
        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
    #
    ###################################################################
    # Features handling
    h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')
    features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
    attn_coord_save_path = os.path.join(r_slide_save_dir, '{}_attn_coords.h5'.format(slide_id))
    logits_coord_save_path = os.path.join(r_slide_save_dir, '{}_logits_coords.h5'.format(slide_id))
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
    #
    features_dir = getattr(data_args, 'features_dir', None)
    provided_features_path = (os.path.join(features_dir, slide_id + '.pt') if features_dir is not None else None)
    use_provided_features = provided_features_path is not None and os.path.isfile(provided_features_path)
    provided_coords = roi_dataset.coords if (use_provided_features and roi_dataset is not None) else None
    if use_provided_features and provided_coords is None and not os.path.isfile(h5_path):
        raise ValueError('features_dir has {}.pt but no coordinates for it: also set data_arguments.coordinates_dir so each embedding row maps to a patch'.format(slide_id))
    ###################################################################
    # Inference
    features, coords, Y_hat, ids, probs, A, patch_logits = run_coarse_inference(wsi_object, img_transforms, feature_extractor, model, model_args, exp_args, h5_path, features_path, attn_coord_save_path, logits_coord_save_path, blocky_wsi_kwargs, roi_dataset=roi_dataset, provided_features_path=provided_features_path, provided_coords=provided_coords)
    preds_str = np.array([reverse_label_dict[idx] for idx in ids])
    print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in probs]))
    ###################################################################
    # Update process stack and save
    process_stack.loc[i, 'bag_size'] = len(features)
    for c in range(exp_args.n_classes):
        process_stack.loc[i, 'Pred_{}'.format(c)] = preds_str[c]
        process_stack.loc[i, 'p_{}'.format(c)] = probs[c]
    #os.makedirs('heatmaps/results/', exist_ok=True)
    if data_args.process_list is not None:
        process_stack.to_csv('{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
    else:
        process_stack.to_csv('{}.csv'.format(exp_args.save_exp_code), index=False)
    ###################################################################
    # Save images, sampled patches, h5 files (attention scores + coords)
    # Save original slide (at vis_level) if requested
    if heatmap_args.save_orig:
        vis_level = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else vis_params['vis_level']
        heatmap_save_name = '{}_orig_{}.{}'.format(slide_id, int(vis_level), heatmap_args.save_ext)
        img_path = os.path.join(p_slide_save_dir, heatmap_save_name)
        save_original_slide(img_path, wsi_object, heatmap_args, vis_level, p_slide_save_dir, slide_id)
    #
    heatmap_nooverlap_path = os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id))
    heatmap_overlap_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))
    wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
    ref_scores = A if heatmap_args.use_ref_scores else None
    wsi_object.saveSegmentation(mask_file)
    #
    # Sample top-k patches
    sample_topk_patches(A, coords, sample_args, wsi_object, label, Y_hat, patch_args, exp_args, slide_id) 
    # Heatmap without overlapping
    draw_blockmap(heatmap_nooverlap_path, A, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, r_slide_save_dir, slide_id)
    #
    # Heatnap with overlapping (slower). Exception if coordinates or features were provided.
    # Calculate the attention scores for the overlapping patches
    if heatmap_args.calc_heatmap and not use_provided_coords and not use_provided_features:
        compute_from_patches(wsi_object=wsi_object, img_transforms=img_transforms, feature_extractor=feature_extractor,
                              model=model, model_type=model_args.model_type, batch_size=exp_args.batch_size,
                              attn_save_path=heatmap_overlap_path, clam_pred=Y_hat, ref_scores=ref_scores, **wsi_kwargs)
    #
    if not os.path.isfile(heatmap_overlap_path):
        print('H5 file with overlap patches (features and coordinates) {} not found'.format(heatmap_overlap_path))
        return
    else:
        with h5py.File(heatmap_overlap_path, 'r') as file:
            fine_scores = file['attention_scores'][:]
            fine_coords = file['coords'][:]
        draw_final_heatmap(fine_scores, fine_coords, slide_path, wsi_object, heatmap_args, vis_patch_size, patch_args, top_left, bot_right, p_slide_save_dir, slide_id)
    



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args.config_file = "/data/pa_cpgarchive/users/Catherine/P8/inference/processlist/config_test.yaml"

# def main():
config_path = args.config_file
config_dict = yaml.safe_load(open(config_path, 'r'))
config_dict = parse_config_dict(args, config_dict)
#
for key, value in config_dict.items():
    if isinstance(value, dict):
        print('\n' + key)
        for value_key, value_value in value.items():
            print(value_key + " : " + str(value_value))
    else:
        print('\n' + key + " : " + str(value))

patch_args, data_args, model_args, encoder_args, exp_args, heatmap_args, sample_args = build_arg_namespaces(config_dict)
#
patch_size = tuple([patch_args.patch_size for _ in range(2)])
step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
    patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))
#
def_seg_params, def_filter_params, def_vis_params, def_patch_params = load_default_params(data_args)
process_stack = build_process_stack(data_args, def_seg_params, def_filter_params, def_vis_params, def_patch_params)

model, feature_extractor, img_transforms = load_model_and_encoder(model_args, encoder_args, load_encoder=encoder_is_needed(process_stack, data_args))
reverse_label_dict = build_reverse_label_dict(data_args)

os.makedirs(exp_args.production_save_dir, exist_ok=True)
os.makedirs(exp_args.raw_save_dir, exist_ok=True)

for i in tqdm(range(len(process_stack))):
    process_slide(i, process_stack, patch_args, data_args, exp_args, heatmap_args, sample_args, model_args,
                    def_seg_params, def_filter_params, def_vis_params, patch_size, step_size,
                    model, feature_extractor, img_transforms, reverse_label_dict)

with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
    yaml.dump(config_dict, outfile, default_flow_style=False)

print('\nDone! All slides processed. Results saved to: {}'.format(os.path.join(exp_args.production_save_dir, exp_args.save_exp_code)))


if __name__ == '__main__':
    main()

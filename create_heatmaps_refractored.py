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
#     overrides the configured patch_size. See build_wsi_from_provided_coords /
#     load_provided_coords.
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


def load_model_and_encoder(model_args, encoder_args):
    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name,
                                                     target_img_size=encoder_args.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
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


def resolve_per_slide_params(row, def_seg_params, def_filter_params, def_vis_params):
    seg_params = def_seg_params.copy()
    filter_params = def_filter_params.copy()
    vis_params = def_vis_params.copy()

    seg_params = load_params(row, seg_params)
    filter_params = load_params(row, filter_params)
    vis_params = load_params(row, vis_params)

    keep_ids = str(seg_params['keep_ids'])
    if len(keep_ids) > 0 and keep_ids != 'none':
        seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
    else:
        seg_params['keep_ids'] = []

    exclude_ids = str(seg_params['exclude_ids'])
    if len(exclude_ids) > 0 and exclude_ids != 'none':
        seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
    else:
        seg_params['exclude_ids'] = []

    for params in (seg_params, filter_params, vis_params):
        for key, val in params.items():
            print('{}: {}'.format(key, val))

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
    Load pre-computed patch coordinates (e.g. produced upstream by slide2vec,
    which already performed its own tissue segmentation) and the patch size to
    read them at. 
    Spcific for slide2vec outputs: When the file's `tile_size_lv0` field is present -- the patch
    size at level 0 -- it takes precedence over the configured patch_size (which
    is only meaningful together with patch_level == 0, since read_region() needs
    a size in the *target* level's pixel units).
    """
    coords_struct = np.load(coord_path)
    coords = np.stack([coords_struct['x'], coords_struct['y']], axis=1).astype(int)

    if coords_struct.dtype.names is not None and 'tile_size_lv0' in coords_struct.dtype.names:
        tile_size = int(coords_struct['tile_size_lv0'][0])
        patch_size = (tile_size, tile_size)
    else:
        patch_size = default_patch_size

    return coords, patch_size


def build_wsi_from_provided_coords(slide_path, coord_path, patch_args, img_transforms, vis_params, mask_path):
    """
    Build the WSI object and its patch dataset directly from a coordinates file
    that already encodes tissue boundaries (post-segmentation), skipping CLAM's
    own tissue segmentation entirely.
    """
    wsi_object = WholeSlideImage(slide_path)
    visualize_and_save_mask(wsi_object, vis_params, mask_path) #TODO: currently saving without using

    default_patch_size = tuple([patch_args.patch_size for _ in range(2)])
    coords, patch_size = load_provided_coords(coord_path, default_patch_size)

    roi_dataset = Wsi_Region_FromCoords(wsi_object, coords=coords, patch_size=patch_size,
                                         level=patch_args.patch_level, t=img_transforms,
                                         custom_downsample=patch_args.custom_downsample)
    return wsi_object, roi_dataset, patch_size


# ---------------------------------------------------------------------------
# Inference (coarse whole-slide pass + patch sampling)
# ---------------------------------------------------------------------------

def run_coarse_inference(wsi_object, img_transforms, feature_extractor, model, model_args, exp_args,
                          h5_path, features_path, block_map_save_path, blocky_wsi_kwargs, roi_dataset=None):
    """
    One inference step for the whole slide: either reuse features cached from a
    previous run (only the cheap model forward is redone, to refresh the
    prediction/attention-score cache) or extract features fresh and run the model
    on the assembled bag in the very same pass, via compute_from_patches().

    `roi_dataset`, when given (e.g. built from a provided coordinates file),
    is used instead of generating patches from tissue contours.
    """
    if os.path.isfile(features_path):
        features = torch.load(features_path)
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
        Y_hat, ids, probs, A = infer_slide(model, features, model_type=model_args.model_type, k=exp_args.n_classes)
        save_hdf5(block_map_save_path, {'attention_scores': A, 'coords': coords}, mode='w')
    else:
        features, coords, Y_hat, ids, probs, A = compute_from_patches(
            wsi_object=wsi_object, img_transforms=img_transforms, feature_extractor=feature_extractor,
            model=model, model_type=model_args.model_type, batch_size=exp_args.batch_size,
            attn_save_path=block_map_save_path, feat_save_path=h5_path, clam_pred=None,
            k=exp_args.n_classes, roi_dataset=roi_dataset, **blocky_wsi_kwargs)
        torch.save(features, features_path)

    return features, coords, Y_hat, ids, probs, A


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

def draw_blockmap(scores, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, r_slide_save_dir, slide_id):
    heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
    if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
        return

    scaling = getattr(heatmap_args, 'scaling', 'percentile')
    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                           alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=heatmap_args.vis_level,
                           blank_canvas=False, thresh=-1, patch_size=vis_patch_size, scaling=scaling)
    heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
    del heatmap


def draw_final_heatmap(scores, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, patch_args,
                        top_left, bot_right, p_slide_save_dir, slide_id):
    # if ref_scores were already used to normalize `scores` upstream, don't rescale again here
    scaling = 'none' if heatmap_args.use_ref_scores else getattr(heatmap_args, 'scaling', 'percentile')

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


def save_original_slide(wsi_object, heatmap_args, vis_params, p_slide_save_dir, slide_id):
    vis_level = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else vis_params['vis_level']
    heatmap_save_name = '{}_orig_{}.{}'.format(slide_id, int(vis_level), heatmap_args.save_ext)
    if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
        return

    heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True,
                                 custom_downsample=heatmap_args.custom_downsample)
    if heatmap_args.save_ext == 'jpg':
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
    else:
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(i, process_stack, patch_args, data_args, exp_args, heatmap_args, sample_args, model_args,
                   def_seg_params, def_filter_params, def_vis_params, patch_size, step_size,
                   model, feature_extractor, img_transforms, reverse_label_dict):
    slide_name, slide_id, label, grouping = resolve_slide_identity(process_stack, i, data_args, reverse_label_dict)
    print('\nprocessing: ', slide_name)
    print('slide id: ', slide_id)

    p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
    os.makedirs(p_slide_save_dir, exist_ok=True)

    r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
    os.makedirs(r_slide_save_dir, exist_ok=True)

    top_left, bot_right = resolve_roi_bounds(heatmap_args, process_stack, i)
    print('top left: ', top_left, ' bot right: ', bot_right)

    slide_path = resolve_slide_path(data_args, process_stack, i, slide_name)

    mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')
    mask_path = os.path.join(r_slide_save_dir, slide_id + '_mask.jpg')

    seg_params, filter_params, vis_params = resolve_per_slide_params(process_stack.loc[i], def_seg_params,
                                                                       def_filter_params, def_vis_params)

    # If patch coordinates were already computed upstream (e.g. by slide2vec, which does its own tissue segmentation), use them directly and skip both CLAM's tissue segmentation and its contour-based patch generation.
    coord_dir = getattr(data_args, 'coord_dir', None)
    coord_path = os.path.join(coord_dir, slide_id + '.npy') if coord_dir is not None else None
    use_provided_coords = coord_path is not None and os.path.isfile(coord_path)

    if use_provided_coords:
        wsi_object, roi_dataset, slide_patch_size = build_wsi_from_provided_coords(
            slide_path, coord_path, patch_args, img_transforms, vis_params, mask_path)
        vis_patch_size = tuple((np.array(slide_patch_size) * patch_args.custom_downsample).astype(int))
    else:
        # If a mask was already produced upstream instead, reuse it and skip only CLAM's segmentation step (patches are still contour-generated).
        provided_mask_dir = getattr(data_args, 'provided_mask_dir', None)
        provided_mask_path = (os.path.join(provided_mask_dir, slide_id + '_mask.pkl')
                              if provided_mask_dir is not None else None)

        wsi_object = segment_and_visualize(slide_path, mask_file, mask_path, seg_params, filter_params, vis_params,
                                            provided_mask_path=provided_mask_path)
        roi_dataset = None

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

    h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')
    features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
    block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))

    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                          'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                          'use_center_shift': heatmap_args.use_center_shift}

    features, coords, Y_hat, ids, probs, A = run_coarse_inference(
        wsi_object, img_transforms, feature_extractor, model, model_args, exp_args,
        h5_path, features_path, block_map_save_path, blocky_wsi_kwargs, roi_dataset=roi_dataset)

    preds_str = np.array([reverse_label_dict[idx] for idx in ids])
    print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label,
                                                 ["{:.4f}".format(p) for p in probs]))

    if not use_provided_coords:
        wsi_object.saveSegmentation(mask_file)

    process_stack.loc[i, 'bag_size'] = len(features)
    for c in range(exp_args.n_classes):
        process_stack.loc[i, 'Pred_{}'.format(c)] = preds_str[c]
        process_stack.loc[i, 'p_{}'.format(c)] = probs[c]

    os.makedirs('heatmaps/results/', exist_ok=True)
    if data_args.process_list is not None:
        process_stack.to_csv('{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
    else:
        process_stack.to_csv('{}.csv'.format(exp_args.save_exp_code), index=False)

    sample_topk_patches(A, coords, sample_args, wsi_object, label, Y_hat, patch_args, exp_args, slide_id)

    draw_blockmap(A, coords, slide_path, wsi_object, heatmap_args, vis_patch_size, r_slide_save_dir, slide_id)

    save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))
    ref_scores = A if heatmap_args.use_ref_scores else None

    wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
                  'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                  'use_center_shift': heatmap_args.use_center_shift}

    if use_provided_coords:
        # There is no separate overlapping patch grid to compute when the patch
        # coordinates were supplied directly (they're already a fixed, tissue-
        # segmented tile set) -- reuse the same per-patch attention/coords
        # computed for the block map instead of a second pass.
        save_hdf5(save_path, {'attention_scores': A, 'coords': coords}, mode='w')
    elif heatmap_args.calc_heatmap:
        compute_from_patches(wsi_object=wsi_object, img_transforms=img_transforms, feature_extractor=feature_extractor,
                              model=model, model_type=model_args.model_type, batch_size=exp_args.batch_size,
                              attn_save_path=save_path, clam_pred=Y_hat, ref_scores=ref_scores, **wsi_kwargs)

    if not os.path.isfile(save_path):
        print('heatmap {} not found'.format(save_path))
        if heatmap_args.use_roi:
            save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
            print('found heatmap for whole slide')
        else:
            return

    if not os.path.isfile(save_path):
        return

    with h5py.File(save_path, 'r') as file:
        fine_scores = file['attention_scores'][:]
        fine_coords = file['coords'][:]

    draw_final_heatmap(fine_scores, fine_coords, slide_path, wsi_object, heatmap_args, vis_patch_size, patch_args,
                        top_left, bot_right, p_slide_save_dir, slide_id)

    if heatmap_args.save_orig:
        save_original_slide(wsi_object, heatmap_args, vis_params, p_slide_save_dir, slide_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = args.config_file
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))

    patch_args, data_args, model_args, encoder_args, exp_args, heatmap_args, sample_args = build_arg_namespaces(config_dict)

    patch_size = tuple([patch_args.patch_size for _ in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
        patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    def_seg_params, def_filter_params, def_vis_params, def_patch_params = load_default_params(data_args)
    process_stack = build_process_stack(data_args, def_seg_params, def_filter_params, def_vis_params, def_patch_params)

    model, feature_extractor, img_transforms = load_model_and_encoder(model_args, encoder_args)
    reverse_label_dict = build_reverse_label_dict(data_args)

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    for i in tqdm(range(len(process_stack))):
        process_slide(i, process_stack, patch_args, data_args, exp_args, heatmap_args, sample_args, model_args,
                       def_seg_params, def_filter_params, def_vis_params, patch_size, step_size,
                       model, feature_extractor, img_transforms, reverse_label_dict)

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    main()

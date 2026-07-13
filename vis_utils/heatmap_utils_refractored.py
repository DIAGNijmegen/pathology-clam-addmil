# Refactor of heatmap_utils.py:
#   - initialize_wsi() gained `provided_mask_path`: if it points to an existing
#     segmentation pickle (the format saveSegmentation() writes), segmentation is
#     skipped and that mask is loaded instead via wsi_object.initSegmentation().
#     Lets inputs that already carry their own tissue mask (e.g. slide2vec) skip
#     CLAM's own Otsu/thresholding segmentation.
#   - drawHeatmap() gained a `scaling` flag: 'percentile' (original behavior),
#     'minmax' (new -- linear rescale by the scores' own min/max, via
#     minmax_scale()), or 'none' (raw passthrough, for scores already normalized
#     upstream against a reference distribution).
#   - The CLAM/addmil branching that used to be duplicated independently in
#     infer_single_slide() and inside compute_from_patches() is now one shared
#     pair of functions: predict_bag() (slide-level prediction) and
#     get_attention_scores() (per-instance attention, given a known or
#     to-be-determined target class), composed by infer_slide() into the single
#     inference step used everywhere a bag of features needs a prediction and/or
#     attention map.
#   - compute_from_patches() now extracts every patch's features in one streamed,
#     batched pass (batching matters only for the raw-image encoder step) and
#     runs the model exactly once on the assembled bag, via infer_slide()/
#     get_attention_scores(). This removes the previous addmil-specific hack of
#     stashing every batch's logits to a temporary h5 file and only computing a
#     real prediction on the last batch (additive MIL's softmax is only
#     meaningful over the whole bag, not a mini-batch of it).
#   - compute_from_patches() also gained an optional `roi_dataset` argument: when
#     given, it's used as-is instead of building a Wsi_Region from `wsi_kwargs`.
#     This lets callers substitute a dataset built from pre-computed coordinates
#     (dataset_modules.wsi_dataset.Wsi_Region_FromCoords) for slides whose tissue
#     contours were never computed (see create_heatmaps_refractored.py's
#     data_arguments.coord_dir handling).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score2percentile(score, ref):
    # https://github.com/mahmoodlab/CLAM/issues/153
    return percentileofscore(ref, score)


def minmax_scale(scores):
    """Alternative to percentile scaling: linearly rescale raw scores to [0, 100] by their own min/max."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo) * 100


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, scaling='percentile', **kwargs):
    """
    scaling controls how raw attention scores are normalized before colormapping:
      - 'percentile': rank scores against each other (original CLAM behavior)
      - 'minmax': linearly rescale between the scores' own min and max
      - 'none': use scores as-is (e.g. when they were already normalized upstream,
                such as against a reference score distribution)
    """
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    if scaling not in ('percentile', 'minmax', 'none'):
        raise ValueError("scaling must be 'percentile', 'minmax' or 'none', got {!r}".format(scaling))

    kwargs.pop('convert_to_percentiles', None)
    if scaling == 'minmax':
        scores = minmax_scale(scores)

    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level,
                                     convert_to_percentiles=(scaling == 'percentile'), **kwargs)
    return heatmap


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None, provided_mask_path=None):
    """
    Build a WholeSlideImage and its tissue segmentation.

    If `provided_mask_path` points to an existing pickle (the same format written by
    saveSegmentation), segmentation is skipped entirely and that mask is loaded
    instead. This lets inputs that already carry their own tissue mask -- e.g.
    slide2vec, which does its own tissue detection upstream -- reuse it instead of
    re-running (and potentially disagreeing with) CLAM's own Otsu/thresholding
    segmentation.
    """
    wsi_object = WholeSlideImage(wsi_path)

    if provided_mask_path is not None and os.path.isfile(provided_mask_path):
        print('Using provided mask, skipping segmentation: ', provided_mask_path)
        wsi_object.initSegmentation(provided_mask_path)
        return wsi_object

    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def get_attention_scores(model, features, model_type=None, clam_pred=None):
    """
    Given an already-extracted bag of patch features, return per-instance attention
    scores as an (N, 1) array.

    Shared by the whole-slide prediction pass and the heatmap-attention pass -- this
    model-specific branching used to be duplicated independently in
    infer_single_slide() and inside compute_from_patches().
    """
    features = features.to(device)
    with torch.inference_mode():
        if model_type == 'addmil':
            logits, att_raw, results_dict = model(features)
            if clam_pred is None:
                clam_pred = torch.topk(logits, 1, dim=1)[1].item()
            patch_logits = results_dict['patch_logits']
            patch_logits = patch_logits * patch_logits.shape[0]
            attention_scores = F.softmax(patch_logits, dim=1).cpu().numpy()[..., clam_pred]
            A = attention_scores.reshape(attention_scores.shape[0], 1)
        else:  # CLAM_SB / CLAM_MB
            A = model(features, attention_only=True)
            if A.size(0) > 1:  # multi-branch: keep the target class's attention row
                assert clam_pred is not None, 'clam_pred is required to select a CLAM_MB attention branch'
                A = A[clam_pred]
            A = A.view(-1, 1).cpu().numpy()
    return A


def predict_bag(model, features, model_type=None, k=1):
    """Full-bag slide-level prediction: predicted class plus top-k class ids/probabilities."""
    features = features.to(device)
    with torch.inference_mode():
        if model_type == 'addmil':
            logits, att_raw, results_dict = model(features)
            Y_hat = torch.topk(logits, 1, dim=1)[1].item()
            Y_prob = F.softmax(logits, dim=1)
        else:  # CLAM_SB / CLAM_MB
            logits, Y_prob, Y_hat, A, results_dict = model(features)
            Y_hat = Y_hat.item()

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
    return Y_hat, ids, probs


def infer_slide(model, features, model_type=None, k=1):
    """
    The single inference step over a whole assembled bag of features: one model
    forward pass' worth of work yields both the slide-level prediction and the
    per-instance attention map.

    This replaces the previous split between infer_single_slide() (which only
    produced the prediction, reading a features.pt that had been separately saved
    and reloaded from disk) and the attention-only logic that used to be
    duplicated inside compute_from_patches().
    """
    Y_hat, ids, probs = predict_bag(model, features, model_type=model_type, k=k)
    A = get_attention_scores(model, features, model_type=model_type, clam_pred=Y_hat)
    return Y_hat, ids, probs, A


def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None,
                          model_type=None, batch_size=512, attn_save_path=None, ref_scores=None,
                          feat_save_path=None, k=1, roi_dataset=None, **wsi_kwargs):
    """
    Stream patches from a WSI region and extract their features in batches (batching
    only matters here, since this is the expensive step over raw images). Once every
    patch's feature vector has been assembled into a single bag, the classification
    model is run exactly once on that bag -- either for attention scores alone (when
    `clam_pred`, the slide-level prediction, is already known) or, when it isn't,
    for both the prediction and its attention map together via infer_slide().

    Previously the per-patch loop ran the model once per mini-batch, and for the
    additive model had to stash every batch's logits to a temporary h5 file and only
    combine them into a real prediction on the very last batch (since additive MIL's
    softmax is only meaningful over the whole bag, not a mini-batch of it).
    Assembling the full bag first removes that batch-order-dependent hack along with
    the redundant per-batch model calls.

    `roi_dataset`, when supplied, is used as-is instead of building a Wsi_Region
    from `wsi_kwargs` -- this lets callers substitute a dataset built from
    pre-computed coordinates (e.g. Wsi_Region_FromCoords) when tissue contours
    were never computed for this slide.
    """
    if roi_dataset is None:
        roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    print('number of batches: ', len(roi_loader))

    all_features = []
    all_coords = []
    with torch.inference_mode():
        for roi, coords in tqdm(roi_loader):
            roi = roi.to(device)
            all_features.append(feature_extractor(roi).cpu())
            all_coords.append(coords.numpy())

    features = torch.cat(all_features, dim=0)
    coords = np.concatenate(all_coords, axis=0)

    if feat_save_path is not None:
        save_hdf5(feat_save_path, {'features': features.numpy(), 'coords': coords}, mode='w')

    Y_hat, ids, probs, A = None, None, None, None
    if model is not None:
        if clam_pred is None:
            Y_hat, ids, probs, A = infer_slide(model, features, model_type=model_type, k=k)
        else:
            A = get_attention_scores(model, features, model_type=model_type, clam_pred=clam_pred)

        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)

        if attn_save_path is not None:
            save_hdf5(attn_save_path, {'attention_scores': A, 'coords': coords}, mode='w')

    return features, coords, Y_hat, ids, probs, A

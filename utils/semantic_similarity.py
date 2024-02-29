import json
import torch
from torch.nn import functional as F
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import clip
import torch.nn as nn
from agents.vlm_bot import build_clip


_MODEL_NAMES_BERT = {
    "bert": "bert-large-uncased",
    "roberta": "roberta-large",
}

_MODEL_NAMES_SENTENCE_BERT = {
    "sbert_mini": "all-MiniLM-L6-v2",
    "sbert_base": "all-mpnet-base-v2",
}

_MODEL_NAMES_CLIP = {
    "clip_b32": "ViT-B/32",
    "clip_b16": "ViT-B/16",
    "clip_l14": "ViT-L/14",
    "clip_l14@336px": "ViT-L/14@336px"
}


def cosine2acc(cosine_similarity):
    return (cosine_similarity + 1) / 2


def compute_semantic_similarity(
        pred_names: list, gt_names: list, prompt=None,
        model: str = 'bert',
        device='cpu',
        device_ids='1'
):
    # prompt raw class names
    if prompt == 'a':
        pred_names = ['a ' + x for x in pred_names]
        gt_names = ['a ' + x for x in gt_names]
    elif prompt == 'photo':
        pred_names = ['a photo of a {}'.format(x) for x in pred_names]
        gt_names = ['a photo of a {}'.format(x) for x in gt_names]
    elif prompt == 'scene':
        pred_names = ['a photo of a {} in the scene'.format(x) for x in pred_names]
        gt_names = ['a photo of a {} in the scene'.format(x) for x in gt_names]
    # else:
    #     pred_names = [x for x in pred_names]
    #     pred_names = [x for x in pred_names]

    if model in _MODEL_NAMES_BERT:
        # BERT-series
        from transformers import AutoTokenizer, AutoModel
        model_name = _MODEL_NAMES_BERT.get(model)
        if not model_name: raise NameError(f"model {model} not found")

        print(f'Loading {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        encoder.eval()

        print(f'Tokenizing...')
        pred_inputs = tokenizer(pred_names, padding=True, return_tensors="pt")
        gt_inputs = tokenizer(gt_names, padding=True, return_tensors="pt")

        print(f'Encoding...')
        with torch.no_grad():
            pred_embedding = encoder(**pred_inputs)
            gt_embedding = encoder(**gt_inputs)

            pred_outputs = pred_embedding.pooler_output
            gt_outputs = gt_embedding.pooler_output

        pred_feats = pred_outputs.detach().to(device)
        gt_feats = gt_outputs.detach().to(device)

        pred_feats = F.normalize(pred_feats, p=2, dim=1)
        gt_feats = F.normalize(gt_feats,  p=2, dim=1)
        del encoder
        # torch.cuda.empty_cache()
    elif model in _MODEL_NAMES_CLIP:
        # import clip
        model_name = _MODEL_NAMES_CLIP.get(model)
        if not model_name: raise NameError(f"model {model} not found")

        print(f'Loading {model_name}...')
        # build VLM model
        if len(device_ids) > 1:
            encoder, preprocess = build_clip(model_name, device, jit=False, parallel=True)
        else:
            encoder, preprocess = build_clip(model_name, device, jit=False, parallel=False)

        # encoder, preprocess = clip.load(model_name, device=device)

        print(f'Tokenizing...')
        pred_inputs = clip.tokenize(pred_names).to(device)
        pred_outputs = clip.tokenize(gt_names).to(device)

        print(f'Encoding...')
        pred_feats = encoder.encode_text(pred_inputs)
        gt_feats = encoder.encode_text(pred_outputs)

        pred_feats = F.normalize(pred_feats, p=2, dim=1)
        gt_feats = F.normalize(gt_feats,  p=2, dim=1)
        del encoder
        # torch.cuda.empty_cache()
    elif model in _MODEL_NAMES_SENTENCE_BERT:
        from sentence_transformers import SentenceTransformer#, util
        model_name = _MODEL_NAMES_SENTENCE_BERT.get(model)
        if not model_name: raise NameError(f"model {model} not found")

        print(f'Loading {model_name}...')
        model = SentenceTransformer(model_name, device=device)

        print(f'Encoding...')
        pred_feats = model.encode(pred_names, convert_to_tensor=True).to(device)
        gt_feats = model.encode(gt_names, convert_to_tensor=True).to(device)

        pred_feats = F.normalize(pred_feats, p=2, dim=1)
        gt_feats = F.normalize(gt_feats,  p=2, dim=1)
        del model
        # torch.cuda.empty_cache()
    else:
        raise NotImplementedError

    print('pred_feats.shape', pred_feats.shape)
    print('gt_feats.shape', gt_feats.shape)

    semantic_scores = torch.einsum('ij, ij->i', pred_feats, gt_feats)
    # semantic_scores = semantic_scores if 'clip' in model else semantic_scores.apply_(cosine2acc)
    ssACC = semantic_scores.mean()

    print(f'Semantic similarity score = {ssACC}')
    return ssACC









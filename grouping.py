import argparse
import torch
import os
from termcolor import colored
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from collections import defaultdict
import clip
from torch.nn import functional as F

from utils.configuration import setup_config, seed_everything
from utils.fileios import *
from utils.metrics import clustering_acc
from utils.semantic_similarity import compute_semantic_similarity


from data import DATA_STATS, DATA_DISCOVERY, DATA_GROUPING, DATA_TRANSFORM
from data.utils import tta, plain_augmentation

from agents.vlm_bot import build_clip
from sklearn.cluster import KMeans, DBSCAN


def wrap_names(cname_list: list):
    try_template = "A photo of a {}, which is a bird."
    new_list = [try_template.format(cname) for cname in cname_list]
    return new_list


def unify_cnames(name: str):
    # new = name.replace('-', ' ')
    # new = new.replace("'s", '')
    new = name.strip()
    # new = new.title()
    return new


def generate_cname_classifier(cfg, encoder_t, gt_cnames: list):
    tokenized_gt_cnames = clip.tokenize(gt_cnames).to(cfg['device'])
    gt_cnames_encoding = F.normalize(encoder_t.encode_text(tokenized_gt_cnames))
    gt_cnames_list = deepcopy(gt_cnames)
    return gt_cnames_encoding, gt_cnames_list, len(gt_cnames_encoding)


def get_tta_exemplars(cfg, examplar, N_tta=10):
    T_plain = plain_augmentation(cfg['image_size'])
    T = tta(cfg['image_size'])

    og_img = T_plain(examplar).unsqueeze(0)
    og_img = og_img.to(cfg['device'])
    anchor_pool = [og_img]

    for i in range(N_tta):
        anchor = T(examplar).unsqueeze(0)
        anchor = anchor.to(cfg['device'])
        anchor_pool.append(anchor)
    return anchor_pool


def generate_voted_classifier(
        cfg,
        encoder,
        guessed_cnames: list,
        modality: str = 'single',
        alpha: float = 0.5,
        N_tta = 0,
        expt_id_suffix = ''
):
    tfms = DATA_TRANSFORM[cfg['dataset_name']](224)
    data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)

    if len(cfg['device_ids']) > 1:
        voting_encoder, _ = build_clip('ViT-L/14', cfg['device'], jit=False, parallel=True)
    else:
        voting_encoder, _ = build_clip('ViT-L/14', cfg['device'], jit=False, parallel=False)

    vote_tokenized_cnames = clip.tokenize(guessed_cnames).to(cfg['device'])
    vote_cnames_encoding = F.normalize(voting_encoder.encode_text(vote_tokenized_cnames))

    if modality == 'single':
        # voting
        candidates_indices = []
        for idx, (img, label) in tqdm(enumerate(data_discovery)):
            img = tfms(img).unsqueeze(0)
            img = img.to(cfg['device'])

            img_encoding = voting_encoder.encode_image(img)
            img_encoding = F.normalize(img_encoding)

            score_clip = img_encoding @ vote_cnames_encoding.T
            idx_top1 = score_clip.argmax(dim=1)
            candidates_indices.append(int(idx_top1[0]))

        # choose the candidates after voting
        candidates_indices = list(set(candidates_indices))
        print(f"Number of selected candidates = {len(candidates_indices)}")
    elif modality == 'cross':
        candidates_pairs = defaultdict(list)
        for idx, (img, label) in tqdm(enumerate(data_discovery)):
            img_vote = tfms(img).unsqueeze(0)
            img_vote = img_vote.to(cfg['device'])

            img_vote_encoding = voting_encoder.encode_image(img_vote)
            img_vote_encoding = F.normalize(img_vote_encoding)

            score_clip = img_vote_encoding @ vote_cnames_encoding.T
            idx_top1 = score_clip.argmax(dim=1)
            idx_top1 = int(idx_top1[0])

            candidates_pairs[idx_top1].extend(
                get_tta_exemplars(cfg, img, N_tta=N_tta)
            )
        print(f"Number of selected candidates = {len(set(list(candidates_pairs.keys())))}")
    else:
        raise NotImplementedError

    del voting_encoder

    tokenized_cnames = clip.tokenize(guessed_cnames).to(cfg['device'])
    cnames_encoding = F.normalize(encoder.encode_text(tokenized_cnames))

    # build final classifier
    if modality == 'single':
        selected_classifier = cnames_encoding[candidates_indices]
        selected_names = [guessed_cnames[i] for i in candidates_indices]
    elif modality == 'cross':
        selected_classifier = []
        for k, v in candidates_pairs.items():
            vec_txt = cnames_encoding[k]
            v = torch.concat(v, dim=0)
            vec_img = encoder.encode_image(v)
            vec_img = F.normalize(vec_img)
            vec_img = vec_img.mean(dim=0)

            vec_mixed = alpha * vec_txt + (1 - alpha) * vec_img

            selected_classifier.append(vec_mixed.view(1, -1))
        selected_classifier = torch.concat(selected_classifier, dim=0)
        selected_names = [guessed_cnames[k] for k, _ in candidates_pairs.items()]
    else:
        raise NotImplementedError

    return selected_classifier, selected_names, len(selected_classifier)


def main_eval(cfg, data_grouping, gt_category_sheet, encoder, classifier, cls_name_list):
    print("---> Evaluating")
    total_preds= np.array([])
    total_labels = np.array([])

    total_pred_names = []
    total_label_names = []
    total_img_paths = []    # for visualization
    for batch_idx, (images, labels, img_paths) in enumerate(tqdm(data_grouping)):
        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)

        similarity = image_encodings @ classifier.T
        # similarity = F.softmax(similarity/0.1, dim=1)
        prediction = similarity.argmax(dim=1)
        names_prediction = [cls_name_list[pred_idx] for pred_idx in prediction]


        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction.cpu().numpy())
        total_pred_names.extend(names_prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())
        names_label = [gt_category_sheet[gt_idx] for gt_idx in labels]
        total_label_names.extend(names_label)
        #       |- image path
        total_img_paths.extend(img_paths)

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)

    del encoder
    # torch.cuda.empty_cache()

    tryout_sacc_model_zoo = ['sbert_base']
    for try_model in tryout_sacc_model_zoo:
        results[f'ssACC_{try_model}'] = compute_semantic_similarity(total_pred_names, total_label_names,
                                                                    model=try_model, device=cfg['device'],
                                                                    device_ids=cfg['device_ids'])
    return results


def kmeans_eval(cfg, data_grouping, encoder, cluster):
    print("---> Evaluating w/ KMeans")
    total_preds= np.array([])
    total_labels = np.array([])

    for batch_idx, (images, labels, _) in enumerate(tqdm(data_grouping)):
        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        image_encodings = image_encodings.cpu().numpy()

        prediction = cluster.predict(image_encodings)

        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)
    return results


def dbscan_eval(cfg, data_grouping, encoder, cluster):
    print("---> Evaluating w/ KMeans")
    total_preds= np.array([])
    total_labels = np.array([])

    for batch_idx, (images, labels, _) in enumerate(tqdm(data_grouping)):
        images = images.to(cfg['device'])
        labels = labels.to(cfg['device'])

        image_encodings = encoder.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        image_encodings = image_encodings.cpu().numpy()

        prediction = cluster.fit_predict(image_encodings)

        ### Record predictions and labels for this batch
        #       |- pred
        total_preds = np.append(total_preds, prediction)
        #       |- label
        total_labels = np.append(total_labels, labels.cpu().numpy())

    results = {}
    results['acc_clustering'], results['nmi_clustering'], results['ari_clustering'] = \
        clustering_acc(total_preds, total_labels)
    return results


def print_results(results: dict, method: str = 'clip'):
    method_name = method.upper()

    print("\n")
    print(colored("=" * 25 + f" {method_name}-based Final Results " + "=" * 25, "yellow"))
    print("\n")
    print(f"[Clustering]")
    print(f"Total {method_name}-based Clustering Acc: {100 * results['acc_clustering']}")
    print(f"Total {method_name}-based Clustering Nmi: {100 * results['nmi_clustering']}")
    print(f"Total {method_name}-based Clustering Ari: {100 * results['ari_clustering']}")
    print("\n")
    print(f"[ssACC (semantic similarity ACC]")
    for try_model in ['sbert_base']:
        print(f"ssACC_{try_model}: {100 * results[f'ssACC_{try_model}']}")
    print(colored("=" * 25 + "          END          " + "=" * 25, "yellow"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grouping', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--config_file_env',
                        type=str,
                        default='./configs/env_machine.yml',
                        help='location of host environment related config file')
    parser.add_argument('--config_file_expt',
                        type=str,
                        default='./configs/expts/bird200_all.yml',
                        help='location of host experiment related config file')
    parser.add_argument('--visualize',
                        type=bool,
                        default=False,
                        help='whether visualize the results')
    # Hyper-parameters
    parser.add_argument('--alpha',
                        type=float,
                        default=0.7)
    parser.add_argument('--N_tta',
                        type=int,
                        default=10)
    # arguments for control experiments
    parser.add_argument('--num_per_category',
                        type=str,
                        default='3',
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'],
                        )
    parser.add_argument('--num_runs',
                        type=int,
                        default=10)


    # init. configuration
    args = parser.parse_args()
    cfg = setup_config(args.config_file_env, args.config_file_expt)
    print(colored(args, 'yellow'))

    # drop the seed
    seed_everything(cfg['seed'])

    expt_id_suffix = f"_{args.num_per_category}"

    device_count = torch.cuda.device_count()
    print("Number of GPUs:", device_count)

    for i in range(device_count):
        print("Device ID:", i, "Device Name:", torch.cuda.get_device_name(i))

    device_ids = [i for i in range(device_count)]
    cfg['device'] = "cuda"
    cfg['device_ids'] = device_ids
    # cfg['device'] = 'cpu'

    # build names
    gt_cnames = DATA_STATS[cfg['dataset_name']]['class_names']
    gt_category_sheet = deepcopy(gt_cnames)
    guessed_cnames = load_json(cfg['path_llm_gussed_names'] + expt_id_suffix)
    guessed_cnames = [unify_cnames(cname) for cname in guessed_cnames]
    print(guessed_cnames)

    # build VLM model
    if len(cfg['device_ids']) > 1:
        encoder, preprocesser = build_clip(cfg['model_size'], cfg['device'], jit=False, parallel=True)
    else:
        encoder, preprocesser = build_clip(cfg['model_size'], cfg['device'], jit=False, parallel=False)

    # build dataloaders
    data_grouping = DATA_GROUPING[cfg['dataset_name']](cfg)


    vilang_cACC = 0.0
    vilang_sACC = 0.0
    for i in range(args.num_runs):
        # generate classifier
        #   |- upper bound
        gt_classifier, gt_name_list, len_gt_classifier = generate_cname_classifier(cfg, encoder, gt_cnames)

        vilang_classifier, vilang_name_list, len_vilang_classifier = generate_voted_classifier(
            cfg, encoder, guessed_cnames, modality='cross', alpha=args.alpha, N_tta=args.N_tta,
            expt_id_suffix=expt_id_suffix,
        )


        print("---> Each Classifier' shapes")
        print(f"\t GT_classifier = {len_gt_classifier}")
        print(f"\t ViLang_guessed = {len_vilang_classifier}")

        # # run the main program
        gt_results = main_eval(cfg, data_grouping, gt_category_sheet, encoder, gt_classifier, gt_name_list)
        vilang_results = main_eval(cfg, data_grouping, gt_category_sheet, encoder, vilang_classifier, vilang_name_list)

        print_results(gt_results, method="UpperBound: CLIP zero-shot")
        print_results(vilang_results, method=f"Ours: ViLangGuessed w/ alpha={args.alpha}, N_tta={args.N_tta}")

        vilang_cACC += vilang_results['acc_clustering']
        vilang_sACC += vilang_results['ssACC_sbert_base']

    vilang_cACC /= args.num_runs
    vilang_sACC /= args.num_runs

    print("\n")
    print(colored("=" * 25 + f" ViLang Final Results of {args.num_runs} runs, w/ {args.num_per_category} imgs per class"
                  + "=" * 25, "yellow"))
    print("\n")
    print(f"[Clustering]")
    print(f"Clustering ACC: {100*vilang_cACC}")
    print(f"Semantic ACC:   {100*vilang_sACC}")
    print(colored("=" * 25 + "          END          " + "=" * 25, "yellow"))

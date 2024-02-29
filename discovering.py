import torch
import os
import argparse
import json.decoder
from tqdm import tqdm
from termcolor import colored
from collections import Counter
from utils.configuration import setup_config, seed_everything
from utils.fileios import dump_json, load_json, dump_txt

from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY
from data.prompt_identify import prompts_howto
from agents.vqa_bot import VQABot
from agents.llm_bot import LLMBot
import re

# Debugging knob
DEBUG = False


def cint2cname(label: int, cname_sheet: list):
    return cname_sheet[label]


def extract_superidentify(cfg, individual_results):
    words = []
    for v in individual_results.values():
        this_word = v.split(' ')[-1]
        words.append(this_word.lower())
    word_counts = Counter(words)

    if cfg['dataset_name'] == 'pet':
        return [super_name for super_name, _ in word_counts.most_common(2)]
    else:
        return [super_name for super_name, _ in word_counts.most_common(1)]


def extract_python_list(text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, text)
    return matches


def trim_result2json(raw_reply: str):
    """
    the raw_answer is a dirty output from LLM following our template.
    this function helps to extract the target JSON content contained in the
    output.
    """
    if raw_reply.find("Output JSON:") >= 0:
        answer = raw_reply.split("Output JSON:")[1].strip()
    else:
        answer = raw_reply.strip()

    if not answer.startswith('{'): answer = '{' + answer

    if not answer.endswith('}'): answer = answer + '}'

    # json_answer = json.loads(answer)
    return answer


def clean_name(name: str):
    name = name.title()
    name = name.replace("-", " ")
    name = name.replace("'s", "")
    return name


def extract_names(gussed_names, clean=True):
    gussed_names = [name.strip() for name in gussed_names]
    if clean:
        gussed_names = [clean_name(name) for name in gussed_names]
    gussed_names = list(set(gussed_names))
    return gussed_names


def how_to_distinguish(bot, prompt):
    reply = bot.infer(prompt, temperature=0.1)
    used_tokens = bot.get_used_tokens()

    print(20*"=")
    print(reply)
    print(20*"=")

    return reply


def main_identify(cfg, bot, data_disco):
    json_super_classes = {}             # img: [attr1, attr2, ..., attrN]

    for idx, (img, label) in tqdm(enumerate(data_disco)):
        # prompt_identify = "Question: What is the main object in this image (choose from: Car, Flower, or Pokemon)? Answer:"
        prompt_identify = "Question: What is the category (car, bird, flower, dog, cat, or Pokemon) of the main object in this image? Answer:"

        reply, trimmed_reply = bot.describe_attribute(img, prompt_identify)
        trimmed_reply = trimmed_reply.lower()
        json_super_classes[str(idx)] = trimmed_reply

        # DEBUG mode
        if DEBUG and idx >= 2:
            break

    return json_super_classes


def main_describe(cfg, bot, data_disco, prompter, cname_sheet):
    # anser containers
    json_attrs = {}             # img: [attr1, attr2, ..., attrN]
    json_llm_prompts = {}       # img: LLM-prompt (has all attrs)

    for idx, (img, label) in tqdm(enumerate(data_disco)):
        if cfg['dataset_name'] == 'pet':
            # first check what is the animal
            pet_prompt = "Questions: What is the animal in this photo (dog or car)? Answer:"
            pet_re, pet_trimmed_re = bot.describe_attribute(img, pet_prompt)
            pet_trimmed_re = pet_trimmed_re.lower()
            # print(pet_trimmed_re)
            if 'dog' in pet_trimmed_re:
                prompter.set_superclass('dog')
            else:   # cat
                prompter.set_superclass('cat')

        # generate attributes and per-attribute prompts for VQA bot
        attrs = prompter.get_attributes()
        attr_prompts = prompter.get_attribute_prompt()
        if len(attrs) != len(attr_prompts):
            raise IndexError("Attribute list should have the same length as attribute prompts")

        print(f"{idx}: label={label}")

        iname = cint2cname(label, cname_sheet)
        iname += f"_{idx}"
        json_attrs[iname] = []

        # describe each attrs
        pair_attr_reply = []    # (attr1: value)
        for attr, p_attr in zip(attrs, attr_prompts):
            re_attr, trimmed_re_attr = bot.describe_attribute(img, p_attr)
            pair_attr_reply.append([attr, trimmed_re_attr])
            json_attrs[iname].append(trimmed_re_attr)

        # generate LLM prompt
        llm_prompt = prompter.get_llm_prompt(pair_attr_reply)
        json_llm_prompts[iname] = llm_prompt

        print(30 * '=')
        print(iname + f" with label {label}")
        print(30 * '=')
        print()
        print(llm_prompt)
        print()
        print('END' + 30 * '=')
        print()

        # DEBUG mode
        if DEBUG and idx >= 2:
            break

    return json_attrs, json_llm_prompts


def main_guess(cfg, bot, reasoning_prompts):
    prompt_list = reasoning_prompts
    replies_raw = {}
    replies_json_to_save = {}

    # LLM inferring
    for i, (key, prompt) in tqdm(enumerate(prompt_list.items())):
        raw_reply = bot.infer(prompt, temperature=0.9)  # use a high temperature for better diversity
        used_tokens = bot.get_used_tokens()

        replies_raw[key] = raw_reply

        print(30 * '=')
        print(f"\t\tinferring [{i}] for {key} used tokens = {used_tokens}")
        print(30 * '=')
        print("Raw----")
        print(raw_reply)
        print()

        jsoned_reply = trim_result2json(raw_reply=raw_reply)

        replies_json_to_save[key] = jsoned_reply

        print("Trimed----")
        print(jsoned_reply)
        print()
        print('END' + 30 * '=')
        print()

        # DEBUG
        if DEBUG and i >= 2:
            break

    print(30 * '=')
    print(f"\t\t Finish Discovering, token consumed {llm_bot.get_used_tokens()}"
          f" = ${bot.get_used_tokens()*0.001*0.002}")
    print(30 * '=')
    print('END' + 30 * '=')
    print()
    return replies_raw, replies_json_to_save


def post_process(cfg, jsoned_replies):
    reply_list = []
    num_of_failures = 0
    # duplicated dict
    for k, v in jsoned_replies.items():
        print(k)
        print(v)
        print()
        print()
        try:
            v_json = json.loads(v)
            reply_list.append(v_json)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for key: {k}")
            num_of_failures += 1
            continue

        # v_json = json.loads(v)
        # reply_list.append(v_json)

    guessed_names = []
    for item in reply_list:
        guessed_names.extend(list(item.keys()))

    guessed_names = extract_names(guessed_names, clean=False)

    if cfg['dataset_name'] in ['pet', 'dog']:
        clean_gussed_names = []
        for aitem in guessed_names:
            clean_gussed_names.extend(aitem.split(','))
        clean_gussed_names = [name.strip() for name in clean_gussed_names]
        guessed_names = clean_gussed_names

    print(30 * '=')
    print(f"\t\t Finished Post-processing")
    print(30 * '=')

    print(f"\t\t ---> total discovered names = {len(guessed_names)}")
    print(guessed_names)
    print()
    print(f"\t\t ---> total discovered names = {len(guessed_names)}")
    print(f"\t\t ---> number of failure entries = {num_of_failures}")

    print('END' + 30 * '=')
    print()
    return guessed_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode',
                        type=str,
                        default='describe',
                        choices=['identify', 'howto', 'describe', 'guess', 'postprocess'],
                        help='operating mode for each stage')
    parser.add_argument('--config_file_env',
                        type=str,
                        default='./configs/env_bravo.yml',
                        help='location of host environment related config file')
    parser.add_argument('--config_file_expt',
                        type=str,
                        default='./configs/expts/bird.yml',
                        help='location of host experiment related config file')
    # arguments for control experiments
    parser.add_argument('--num_per_category',
                        type=int,
                        default=3)


    args = parser.parse_args()
    print(colored(args, 'blue'))

    cfg = setup_config(args.config_file_env, args.config_file_expt)
    print(colored(cfg, 'yellow'))

    # drop the seed
    seed_everything(cfg['seed'])

    expt_id_suffix = f"_{args.num_per_category}"

    if args.mode == 'identify':
        """
        identify the super-categories of the dataset
        """
        # build VQA Bot
        if cfg['host'] in ["chaos", "YOUR_GPU_CLUSTER_NAME"]:
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cuda', device_id=cfg['device_id'], bit8=False)
        else:
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cpu')

        # get data ordered class name lookup
        cname_sheet = DATA_STATS[cfg['dataset_name']]['class_names']

        # build data set
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)

        # paths to save per-image VQAbot answers (about attributes) and per-image LLM prompts
        save_path_identify_answers = cfg['path_identify_answers'] + expt_id_suffix

        # run the main program to describe the per-img attributes
        superclass_results = main_identify(cfg, vqa_bot, data_discovery)

        identified_super_class = extract_superidentify(cfg, superclass_results)

        print(identified_super_class)
        dump_json(save_path_identify_answers, {'superclass': identified_super_class})
        print(f"Succ. dumped identified super-class values to {save_path_identify_answers}")
    elif args.mode == 'howto':
        """
        consult LLMs about how to describe XXX (e.g., birds)
        NOTE THAT: this step is performed multiple times to get as many as possible useful attributes
        then the attributes are added to the dataset class code, for the ease of auto processing later.
        """
        save_path_vqa_questions = cfg['path_vqa_questions']
        llm_bot = LLMBot(model=cfg['model_type_llm'], temperature=0.1)

        superclass = load_json(cfg['path_identify_answers'] + expt_id_suffix)['superclass']

        if len(superclass) > 1:
            prompt = [
                prompts_howto["pet"].replace('[__SUPERCLASS__]', superclass[0]),
                prompts_howto["pet"].replace('[__SUPERCLASS__]', superclass[1])
            ]
        else:
            if 'bird' in superclass[0]:
                prompt = [prompts_howto['bird'].replace('[__SUPERCLASS__]', 'bird')]
            elif 'car' in superclass[0]:
                prompt = [prompts_howto['car'].replace('[__SUPERCLASS__]', 'car')]
            elif 'dog' in superclass[0]:
                prompt = [prompts_howto['dog'].replace('[__SUPERCLASS__]', 'dog')]
            elif 'flower' in superclass[0]:
                prompt = [prompts_howto['flower'].replace('[__SUPERCLASS__]', 'flower')]
            elif 'pokemon' in superclass[0]:
                prompt = [prompts_howto['pokemon'].replace('[__SUPERCLASS__]', 'pokemon')]

        pattern = r'\[([^\]]*)\]'
        for i, ppt in enumerate(prompt):
            print(ppt)
            vqa_questions = how_to_distinguish(llm_bot, prompt=ppt)
            matches = re.findall(pattern, vqa_questions)
            result = matches[0].strip().replace('\n', '').replace('"', "'").replace("', '", "','")

            if cfg['dataset_name'] == 'pet':
                dump_txt(save_path_vqa_questions.replace('pet_vqa_questions',
                                                         f'pet_{superclass[i]}_vqa_questions.txt'), f'[{result}]')
            else:
                dump_txt(save_path_vqa_questions, f'[{result}]')
    elif args.mode == 'describe':
        """
        describe the attributes
        """
        # build VQA Bot
        if cfg['host'] in ["chaos", "YOUR_GPU_CLUSTER_NAME"]:
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cuda', device_id=cfg['device_id'], bit8=False)
        else:
            vqa_bot = VQABot(model_tag=cfg['model_size_vqa'], device='cpu')

        # get data ordered class name lookup
        cname_sheet = DATA_STATS[cfg['dataset_name']]['class_names']

        # build data set
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)

        # build VQAbot prompter
        prompter = PROMPTERS[cfg['dataset_name']](cfg)

        # paths to save per-image VQAbot answers (about attributes) and per-image LLM prompts
        save_path_vqa_answers = cfg['path_vqa_answers'] + expt_id_suffix
        save_path_llm_prompts = cfg['path_llm_prompts'] + expt_id_suffix

        # run the main program to describe the per-img attributes
        json_vqa_answers, json_llm_prompts = main_describe(cfg, vqa_bot, data_discovery, prompter, cname_sheet)

        dump_json(save_path_vqa_answers, json_vqa_answers)
        print(f"Succ. dumped attribute values to {save_path_vqa_answers}")

        dump_json(save_path_llm_prompts, json_llm_prompts)
        print(f"Succ. dumped LLM prompts  to {save_path_llm_prompts}")
    elif args.mode == 'guess':
        """
        reason category names based on the attribute-description pairs
        """
        reasoning_prompts = load_json(cfg['path_llm_prompts'] + expt_id_suffix)
        llm_bot = LLMBot(model=cfg['model_type_llm'])

        # run the main program
        raw_replies, jsoned_replies = main_guess(cfg, llm_bot, reasoning_prompts)

        # save LLM replis
        dump_json(cfg['path_llm_replies_raw'] + expt_id_suffix, raw_replies)
        dump_json(cfg['path_llm_replies_jsoned'] + expt_id_suffix, jsoned_replies)
    elif args.mode == 'postprocess':
        """
        clean the results a bit
        """
        # load replies
        jsoned_replies = load_json(cfg['path_llm_replies_jsoned'] + expt_id_suffix)
        # post-process data
        gussed_names = post_process(cfg, jsoned_replies)
        # save LLM gussed names
        dump_json(cfg['path_llm_gussed_names'] + expt_id_suffix, gussed_names)
    else:
        raise NotImplementedError


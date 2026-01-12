"""
Unified Belebele Activation Patching Script
Combines main and control experiments in a modular framework

Supports:
- Main experiment: Same sample cross-lingual patching
- Control experiment: Different sample, same answer position
- Single token patching: Patch at specific positions (lasttoken, beforecolon, endofoption)
- Multi token patching: Patch at structurally aligned positions (P, Q, A, B, C, D)
"""

import nnsight
import torch
import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from nnsight import CONFIG
from nnsight import LanguageModel
import json
from collections import defaultdict
import gc
from abc import ABC, abstractmethod

# Environment setup
NDIF_API = os.environ.get('NDIF_API_KEY')
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_HOME = os.environ.get('HF_HOME')
CONFIG.set_default_api_key(NDIF_API)
from datasets import load_dataset


# ============================================================================
# Utility Functions
# ============================================================================

def load_json_object(path):
    """Load JSONL file with evaluation results"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def find_index_of_subarray(data_list, subarray):
    """
    Find the starting index of first occurrence of subarray in list

    Args:
        data_list: List of token IDs
        subarray: Sequence to search for

    Returns:
        Starting index if found, None otherwise
    """
    len_subarray = len(subarray)
    if len_subarray == 0:
        return None

    for i in range(len(data_list) - len_subarray + 1):
        if data_list[i:i + len_subarray] == subarray:
            return i
    return None


def get_ids(args):
    """
    Identify samples for patching based on evaluation results

    Returns indices where English is correct and target language is wrong
    (or vice versa depending on patching direction)
    """
    english_results_path = args.results_path + 'belebele_eval_Llama3.1_eng_Latn_ABCD.jsonl'
    xx_results_path = args.results_path + f'belebele_eval_Llama3.1_{args.lang}_ABCD.jsonl'

    english_eval = load_json_object(english_results_path)
    xx_eval = load_json_object(xx_results_path)

    if args.patching_direction == 'ECtoXW' or args.patching_direction == 'XWtoEC':
        idx = [i for i in range(len(english_eval))
               if english_eval[i]['acc'] == 1.0 and xx_eval[i]['acc'] == 0.0]

    return idx


def create_prompt(dataset, split, mc_val):
    """
    Create prompts for Belebele reading comprehension task

    Args:
        dataset: HuggingFace dataset
        split: Dataset split name
        mc_val: Multiple choice format (e.g., ['A: ', 'B: ', 'C: ', 'D: '])

    Returns:
        Tuple of (prompt_list, answer_list)
    """
    prefix_text = "Based on the below passage and the question, choose the best option. \n\n"

    passage = [dataset[split][i]['flores_passage'] for i in range(len(dataset[split]))]
    question = [dataset[split][i]['question'] for i in range(len(dataset[split]))]
    choice1 = [dataset[split][i]['mc_answer1'] for i in range(len(dataset[split]))]
    choice2 = [dataset[split][i]['mc_answer2'] for i in range(len(dataset[split]))]
    choice3 = [dataset[split][i]['mc_answer3'] for i in range(len(dataset[split]))]
    choice4 = [dataset[split][i]['mc_answer4'] for i in range(len(dataset[split]))]
    answer = [dataset[split][i]['correct_answer_num'] for i in range(len(dataset[split]))]

    prompt = [
        prefix_text + "P: " + passage[i] + "\n" + "Q: " + question[i] + "\n" +
        mc_val[0] + choice1[i] + "\n" +
        mc_val[1] + choice2[i] + "\n" +
        mc_val[2] + choice3[i] + "\n" +
        mc_val[3] + choice4[i] + "\n" +
        "\nAnswer: "
        for i in range(len(dataset[split]))
    ]

    return prompt, answer


def get_target_tokens(llm, mc_val):
    """Extract token IDs for multiple choice options"""
    if mc_val == ['A: ', 'B: ', 'C: ', 'D: ']:
        token_id = [llm.tokenizer.encode(item)[-1] for item in ["A", "B", "C", "D"]]
    elif mc_val == ['1. ', '2. ', '3. ', '4. ']:
        token_id = [llm.tokenizer.encode(item)[-1] for item in ["1", "2", "3", "4"]]
    elif mc_val == ['Q. ', 'Z. ', 'R. ', 'X. ']:
        token_id = [llm.tokenizer.encode(item)[-1] for item in ["Q", "Z", "R", "X"]]
    else:
        raise ValueError(f"Unsupported mc_val format: {mc_val}")

    return token_id


def load_hf_data(args, llm):
    """Load datasets and prepare prompts for both languages"""
    MODE_DICT = {
        "ABCD": ['A: ', 'B: ', 'C: ', 'D: '],
        "1234": ['1. ', '2. ', '3. ', '4. '],
        "QZRX": ['Q. ', 'Z. ', 'R. ', 'X. ']
    }

    eng_mc_val = MODE_DICT[args.eng_mc_key]
    xx_mc_val = MODE_DICT[args.xx_mc_key]

    # Get English data
    eng_dataset = load_dataset("Kartik221/Belebele_test", 'eng_Latn')
    eng_prompt_list, eng_answer = create_prompt(eng_dataset, 'test', eng_mc_val)
    eng_target_tokens = get_target_tokens(llm, eng_mc_val)

    # Get target language data
    xx_dataset = load_dataset("Kartik221/Belebele_test", args.lang)
    xx_prompt_list, xx_answer = create_prompt(xx_dataset, 'test', xx_mc_val)
    xx_target_tokens = get_target_tokens(llm, xx_mc_val)

    return eng_prompt_list, eng_answer, eng_target_tokens, xx_prompt_list, xx_answer, xx_target_tokens


# ============================================================================
# Prompt Selection Strategy
# ============================================================================

class PromptSelector(ABC):
    """Abstract base class for English prompt selection strategies"""

    @abstractmethod
    def select_prompt(self, sample_idx, eng_prompt_list, xx_answer,
                     control_idx_by_answer=None):
        """
        Select which English prompt to use for patching

        Args:
            sample_idx: Index of current target language sample
            eng_prompt_list: List of all English prompts
            xx_answer: Target language answers
            control_idx_by_answer: Dict mapping answers to valid indices (for control)

        Returns:
            Selected English prompt string
        """
        pass


class MainPromptSelector(PromptSelector):
    """Main experiment: Use same sample index for English and target language"""

    def select_prompt(self, sample_idx, eng_prompt_list, xx_answer,
                     control_idx_by_answer=None):
        return eng_prompt_list[sample_idx]


class ControlPromptSelector(PromptSelector):
    """
    Control experiment: Use different English sample with same answer position

    This controls for answer position bias by ensuring the patched activations
    come from a different question but with the same correct answer position.
    """

    def __init__(self, seed=42):
        random.seed(seed)
        self.seed = seed

    def select_prompt(self, sample_idx, eng_prompt_list, xx_answer,
                     control_idx_by_answer=None):
        if control_idx_by_answer is None:
            raise ValueError("Control experiment requires control_idx_by_answer")

        answer = str(xx_answer[sample_idx])

        # Get valid indices (same answer, different sample)
        valid_indices = [i for i in control_idx_by_answer[answer] if i != sample_idx]

        if not valid_indices:
            raise ValueError(f"No valid control samples for answer {answer} at index {sample_idx}")

        # Randomly select one
        selected_idx = random.sample(valid_indices, 1)[0]
        return eng_prompt_list[selected_idx]


def get_prompt_selector(experiment_type, seed=42):
    """Factory function to create appropriate prompt selector"""
    if experiment_type == 'main':
        return MainPromptSelector()
    elif experiment_type == 'control':
        return ControlPromptSelector(seed=seed)
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")


# ============================================================================
# Single Token Patching
# ============================================================================

def run_patching_singletoken(args):
    """
    Patch activations at a single token position

    Supported positions:
    - lasttoken: -1 (final token)
    - beforecolon: -2 (token before colon in "Answer:")
    - endofoption: -3 (end of last option)
    """
    LLM_DICT = {'Llama3.1': 'meta-llama/Meta-Llama-3.1-8B'}
    llm = LanguageModel(LLM_DICT[args.llm_name], dispatch=True, device_map="auto")

    # Map patching position to token index
    position_map = {
        'lasttoken': -1,
        'beforecolon': -2,
        'endofoption': -3
    }
    token_position = position_map[args.patching_position]

    # Load data
    eng_prompt_list, eng_answer, eng_target_tokens, xx_prompt_list, xx_answer, xx_target_tokens = load_hf_data(args, llm)

    # Get sample indices to patch
    idx = get_ids(args)
    if args.start_idx is not None and args.N is not None:
        idx = idx[args.start_idx:args.start_idx + args.N]

    # Initialize prompt selector
    prompt_selector = get_prompt_selector(args.experiment_type, args.seed)

    # Prepare control indices if needed
    control_idx_by_answer = None
    if args.experiment_type == 'control':
        control_idx_by_answer = {
            '1': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '1'],
            '2': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '2'],
            '3': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '3'],
            '4': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '4']
        }

    logits_results = defaultdict(dict)

    # Create output filename
    experiment_prefix = 'control_' if args.experiment_type == 'control' else ''
    json_name = (f'{args.save_path}/{args.component}/{args.patching_position}/'
                f'belebele_{experiment_prefix}eval_{args.llm_name}_{args.patching_direction}_'
                f'{args.patching_position}_eng_Latn_{args.eng_mc_key}_{args.lang}_{args.xx_mc_key}.jsonl')

    # Ensure directory exists
    os.makedirs(os.path.dirname(json_name), exist_ok=True)

    print(f"Running {args.experiment_type} experiment with {args.patching_position} patching")
    print(f"Processing {len(idx)} samples")
    print(f"Results will be saved to: {json_name}")

    if args.patching_direction in ['ECtoXW', 'EWtoXC']:
        for i, sample_idx in enumerate(idx):
            # Select English prompt based on experiment type
            eng_prompt = prompt_selector.select_prompt(
                sample_idx, eng_prompt_list, xx_answer, control_idx_by_answer
            )
            xx_prompt = xx_prompt_list[sample_idx]

            sample_logits = {}

            for layer_idx in tqdm(range(32), desc=f"Sample {i+1}/{len(idx)} (idx={sample_idx})"):
                with torch.no_grad():
                    with llm.session() as session:
                        # Extract activations from English prompt
                        with llm.trace(eng_prompt) as t1:
                            attn_activations = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position,].save()
                            mlp_activations = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position,].save()
                            resid_activations = llm.model.layers[layer_idx].output.squeeze()[token_position,].save()

                        # Patch into target language prompt
                        with llm.trace(xx_prompt) as t2:
                            if args.component == 'attnoutput':
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position,] = attn_activations
                            elif args.component == 'mlpoutput':
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position,] = mlp_activations
                            elif args.component == 'layeroutput':
                                llm.model.layers[layer_idx].output.squeeze()[token_position,] = resid_activations
                            elif args.component == 'all':
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position,] = attn_activations
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position,] = mlp_activations
                                llm.model.layers[layer_idx].output.squeeze()[token_position,] = resid_activations

                            logits = llm.output['logits']
                            last_token_logits = logits[:, -1, :]
                            patched_logits = last_token_logits[:, xx_target_tokens].save()

                    patched_logits = patched_logits[-1].detach().cpu()
                    logits_results[sample_idx][layer_idx] = patched_logits

                    # Memory cleanup
                    torch.cuda.empty_cache()
                    del patched_logits, resid_activations, attn_activations, mlp_activations
                    gc.collect()

                # Save results incrementally
                with open(json_name, "a") as f:
                    for sample_key, layers_v in logits_results.items():
                        if sample_key == sample_idx:
                            for layer_key, patched_logits in layers_v.items():
                                if layer_key == layer_idx:
                                    entry = {
                                        "sample_idx": sample_idx,
                                        "layer_idx": layer_idx,
                                        "logits": patched_logits.tolist()
                                    }
                                    f.write(json.dumps(entry) + "\n")


# ============================================================================
# Multi Token Patching
# ============================================================================

def run_patching_multitoken(args):
    """
    Patch activations at multiple structurally aligned positions

    Patches at positions after:
    - P: (Passage)
    - Q: (Question)
    - A: (Option A)
    - B: (Option B)
    - C: (Option C)
    - D: (Option D / Answer:)
    """
    LLM_DICT = {'Llama3.1': 'meta-llama/Meta-Llama-3.1-8B'}
    llm = LanguageModel(LLM_DICT[args.llm_name], dispatch=True, device_map="auto")

    # Load data
    eng_prompt_list, eng_answer, eng_target_tokens, xx_prompt_list, xx_answer, xx_target_tokens = load_hf_data(args, llm)

    # Get sample indices
    idx = get_ids(args)
    if args.start_idx is not None and args.N is not None:
        idx = idx[args.start_idx:args.start_idx + args.N]

    # Initialize prompt selector
    prompt_selector = get_prompt_selector(args.experiment_type, args.seed)

    # Prepare control indices if needed
    control_idx_by_answer = None
    if args.experiment_type == 'control':
        control_idx_by_answer = {
            '1': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '1'],
            '2': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '2'],
            '3': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '3'],
            '4': [i for i, answer in enumerate(eng_answer) if i in idx and str(answer) == '4']
        }

    logits_results = defaultdict(dict)

    # Create output filename
    experiment_prefix = 'control_' if args.experiment_type == 'control' else ''
    json_name = (f'{args.save_path}/{args.component}/{args.patching_position}/'
                f'belebele_{experiment_prefix}eval_{args.llm_name}_{args.patching_direction}_'
                f'{args.patching_position}_eng_Latn_{args.eng_mc_key}_{args.lang}_{args.xx_mc_key}.jsonl')

    os.makedirs(os.path.dirname(json_name), exist_ok=True)

    print(f"Running {args.experiment_type} experiment with multitoken patching")
    print(f"Processing {len(idx)} samples")
    print(f"Results will be saved to: {json_name}")

    if args.patching_direction in ['ECtoXW', 'EWtoXC']:
        for i, sample_idx in enumerate(idx):
            # Select English prompt
            eng_prompt = prompt_selector.select_prompt(
                sample_idx, eng_prompt_list, xx_answer, control_idx_by_answer
            )
            xx_prompt = xx_prompt_list[sample_idx]

            sample_logits = {}

            for layer_idx in tqdm(range(32), desc=f"Sample {i+1}/{len(idx)} (idx={sample_idx})"):
                with torch.no_grad():
                    with llm.session() as session:
                        # Extract activations from English at multiple positions
                        with llm.trace(eng_prompt) as t1:
                            eng_tokens = llm.tokenizer.encode(eng_prompt)

                            # Find token positions
                            token_position_after_premise = find_index_of_subarray(eng_tokens, [48, 25])
                            token_position_after_question = find_index_of_subarray(eng_tokens, [32, 25])
                            token_position_after_optionA = find_index_of_subarray(eng_tokens, [33, 25])
                            token_position_after_optionB = find_index_of_subarray(eng_tokens, [34, 25])
                            token_position_after_optionC = find_index_of_subarray(eng_tokens, [35, 25])
                            token_position_after_optionD = find_index_of_subarray(eng_tokens, [16533, 25])

                            # Extract activations based on component type
                            if args.component == 'layeroutput':
                                resid_activations_after_premise = llm.model.layers[layer_idx].output.squeeze()[token_position_after_premise-1,].save()
                                resid_activations_after_question = llm.model.layers[layer_idx].output.squeeze()[token_position_after_question-1,].save()
                                resid_activations_after_optionA = llm.model.layers[layer_idx].output.squeeze()[token_position_after_optionA-1,].save()
                                resid_activations_after_optionB = llm.model.layers[layer_idx].output.squeeze()[token_position_after_optionB-1,].save()
                                resid_activations_after_optionC = llm.model.layers[layer_idx].output.squeeze()[token_position_after_optionC-1,].save()
                                resid_activations_after_optionD = llm.model.layers[layer_idx].output.squeeze()[token_position_after_optionD-1,].save()

                            if args.component == 'attnoutput':
                                attn_activations_after_premise = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_premise-1,].save()
                                attn_activations_after_question = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_question-1,].save()
                                attn_activations_after_optionA = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_optionA-1,].save()
                                attn_activations_after_optionB = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_optionB-1,].save()
                                attn_activations_after_optionC = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_optionC-1,].save()
                                attn_activations_after_optionD = llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[token_position_after_optionD-1,].save()

                            if args.component == 'mlpoutput':
                                mlp_activations_after_premise = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_premise-1,].save()
                                mlp_activations_after_question = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_question-1,].save()
                                mlp_activations_after_optionA = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_optionA-1,].save()
                                mlp_activations_after_optionB = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_optionB-1,].save()
                                mlp_activations_after_optionC = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_optionC-1,].save()
                                mlp_activations_after_optionD = llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[token_position_after_optionD-1,].save()

                        # Patch into target language
                        with llm.trace(xx_prompt) as t2:
                            xx_tokens = llm.tokenizer.encode(xx_prompt)

                            # Find corresponding positions in target language
                            xx_token_position_after_premise = find_index_of_subarray(xx_tokens, [48, 25])
                            xx_token_position_after_question = find_index_of_subarray(xx_tokens, [32, 25])
                            xx_token_position_after_optionA = find_index_of_subarray(xx_tokens, [33, 25])
                            xx_token_position_after_optionB = find_index_of_subarray(xx_tokens, [34, 25])
                            xx_token_position_after_optionC = find_index_of_subarray(xx_tokens, [35, 25])
                            xx_token_position_after_optionD = find_index_of_subarray(xx_tokens, [16533, 25])

                            # Patch activations
                            if args.component == 'layeroutput':
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_premise-1,] = resid_activations_after_premise
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_question-1,] = resid_activations_after_question
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_optionA-1,] = resid_activations_after_optionA
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_optionB-1,] = resid_activations_after_optionB
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_optionC-1,] = resid_activations_after_optionC
                                llm.model.layers[layer_idx].output.squeeze()[xx_token_position_after_optionD-1,] = resid_activations_after_optionD

                            if args.component == 'attnoutput':
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_premise-1,] = attn_activations_after_premise
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_question-1,] = attn_activations_after_question
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_optionA-1,] = attn_activations_after_optionA
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_optionB-1,] = attn_activations_after_optionB
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_optionC-1,] = attn_activations_after_optionC
                                llm.model.layers[layer_idx].self_attn.o_proj.output.squeeze()[xx_token_position_after_optionD-1,] = attn_activations_after_optionD

                            if args.component == 'mlpoutput':
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_premise-1,] = mlp_activations_after_premise
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_question-1,] = mlp_activations_after_question
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_optionA-1,] = mlp_activations_after_optionA
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_optionB-1,] = mlp_activations_after_optionB
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_optionC-1,] = mlp_activations_after_optionC
                                llm.model.layers[layer_idx].mlp.down_proj.output.squeeze()[xx_token_position_after_optionD-1,] = mlp_activations_after_optionD

                            logits = llm.output['logits']
                            last_token_logits = logits[:, -1, :]
                            patched_logits = last_token_logits[:, xx_target_tokens].save()

                    patched_logits = patched_logits[-1].detach().cpu()
                    logits_results[sample_idx][layer_idx] = patched_logits

                    # Memory cleanup
                    torch.cuda.empty_cache()
                    del patched_logits

                    if args.component == 'layeroutput':
                        del resid_activations_after_premise, resid_activations_after_question
                        del resid_activations_after_optionA, resid_activations_after_optionB
                        del resid_activations_after_optionC, resid_activations_after_optionD
                    if args.component == 'attnoutput':
                        del attn_activations_after_premise, attn_activations_after_question
                        del attn_activations_after_optionA, attn_activations_after_optionB
                        del attn_activations_after_optionC, attn_activations_after_optionD
                    if args.component == 'mlpoutput':
                        del mlp_activations_after_premise, mlp_activations_after_question
                        del mlp_activations_after_optionA, mlp_activations_after_optionB
                        del mlp_activations_after_optionC, mlp_activations_after_optionD

                    gc.collect()

                # Save results incrementally
                with open(json_name, "a") as f:
                    for sample_key, layers_v in logits_results.items():
                        if sample_key == sample_idx:
                            for layer_key, patched_logits in layers_v.items():
                                if layer_key == layer_idx:
                                    entry = {
                                        "sample_idx": sample_idx,
                                        "layer_idx": layer_idx,
                                        "logits": patched_logits.tolist()
                                    }
                                    f.write(json.dumps(entry) + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Belebele Activation Patching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Main experiment with single token patching
  python belebele_patching_unified.py --experiment_type main --patching_mode singletoken \\
    --patching_position lasttoken --component layeroutput --lang zho_Hans

  # Control experiment with single token patching
  python belebele_patching_unified.py --experiment_type control --patching_mode singletoken \\
    --patching_position lasttoken --component layeroutput --lang zho_Hans --seed 42

  # Main experiment with multi token patching
  python belebele_patching_unified.py --experiment_type main --patching_mode multitoken \\
    --component layeroutput --lang zho_Hans --start_idx 0 --N 50
        """
    )

    # Experiment configuration
    parser.add_argument('--experiment_type', type=str, required=True,
                       choices=['main', 'control'],
                       help='Type of experiment: main (same sample) or control (different sample, same answer)')
    parser.add_argument('--patching_mode', type=str, required=True,
                       choices=['singletoken', 'multitoken'],
                       help='Patching mode: singletoken or multitoken')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for control experiment (default: 42)')

    # Model configuration
    parser.add_argument('--llm_name', type=str, default='Llama3.1',
                       help='Model name (default: Llama3.1)')

    # Language configuration
    parser.add_argument('--lang', type=str, required=True,
                       help='Target language code (e.g., zho_Hans)')
    parser.add_argument('--eng_mc_key', type=str, default='ABCD',
                       choices=['ABCD', '1234', 'QZRX'],
                       help='English multiple choice format')
    parser.add_argument('--xx_mc_key', type=str, default='ABCD',
                       choices=['ABCD', '1234', 'QZRX'],
                       help='Target language multiple choice format')

    # Patching configuration
    parser.add_argument('--component', type=str, required=True,
                       choices=['attnoutput', 'mlpoutput', 'layeroutput', 'all'],
                       help='Component to patch')
    parser.add_argument('--patching_position', type=str, default='lasttoken',
                       choices=['lasttoken', 'beforecolon', 'endofoption'],
                       help='Token position for single token patching (ignored for multitoken)')
    parser.add_argument('--patching_direction', type=str, default='ECtoXW',
                       choices=['ECtoXW', 'XWtoEC', 'EWtoXC', 'XCtoEW'],
                       help='Patching direction')

    # Data configuration
    parser.add_argument('--results_path', type=str,
                       help='Path to evaluation results')
    parser.add_argument('--save_path', type=str,
                       help='Base path for saving results')

    # Sample selection
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Start index for sample processing')
    parser.add_argument('--N', type=int, default=None,
                       help='Number of samples to process')

    args = parser.parse_args()

    # Validate arguments
    if args.patching_mode == 'multitoken' and args.component == 'all':
        raise ValueError("component='all' is not supported in multitoken mode. Please use 'attnoutput', 'mlpoutput', or 'layeroutput'.")

    # Dispatch to appropriate function
    if args.patching_mode == 'singletoken':
        run_patching_singletoken(args)
    elif args.patching_mode == 'multitoken':
        run_patching_multitoken(args)
    else:
        raise ValueError(f"Unknown patching_mode: {args.patching_mode}")

    print("Patching complete!")


if __name__ == "__main__":
    main()

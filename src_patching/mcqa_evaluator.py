"""
Modular MCQA Benchmark Evaluator
Supports: Belebele, XCOPA, XStoryCloze
"""
import torch
import os
import argparse
import json
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Tuple, Dict


# ============================================================================
# Dataset Configuration
# ============================================================================

class DatasetConfig:
    """Configuration for a specific benchmark dataset"""
    def __init__(self, name: str, hf_path: str, split: str, num_choices: int, answer_offset: int):
        self.name = name
        self.hf_path = hf_path
        self.split = split
        self.num_choices = num_choices
        self.answer_offset = answer_offset  # -1 for 1-indexed answers, 0 for 0-indexed


# Dataset configurations
DATASET_CONFIGS = {
    'belebele': DatasetConfig(
        name='belebele',
        hf_path='Kartik221/Belebele_test',
        split='test',
        num_choices=4,
        answer_offset=-1  # Answers are 1-4, need to convert to 0-3
    ),
    'xcopa': DatasetConfig(
        name='xcopa',
        hf_path='cambridgeltl/xcopa',
        split='test',
        num_choices=2,
        answer_offset=0  # Answers are already 0-1
    ),
    'xstorycloze': DatasetConfig(
        name='xstorycloze',
        hf_path='juletxara/xstory_cloze',
        split='eval',
        num_choices=2,
        answer_offset=-1  # Answers are 1-2, need to convert to 0-1
    )
}


# ============================================================================
# Multiple Choice Format Configurations
# ============================================================================

MC_FORMAT_CONFIGS = {
    2: {  # Binary choice datasets
        'AB': ['A: ', 'B: '],
        '12': ['1. ', '2. '],
        'QZ': ['Q. ', 'Z. ']
    },
    4: {  # Four-choice datasets
        'ABCD': ['A: ', 'B: ', 'C: ', 'D: '],
        '1234': ['1. ', '2. ', '3. ', '4. '],
        'QZRX': ['Q. ', 'Z. ', 'R. ', 'X. ']
    }
}


# ============================================================================
# Prompt Builders
# ============================================================================

class PromptBuilder(ABC):
    """Abstract base class for building dataset-specific prompts"""

    @abstractmethod
    def build_prompts(self, dataset, split: str, mc_val: List[str]) -> Tuple[List[str], List[int]]:
        """
        Build prompts and extract answers from dataset

        Args:
            dataset: HuggingFace dataset object
            split: Dataset split name
            mc_val: Multiple choice format list (e.g., ['A: ', 'B: '])

        Returns:
            Tuple of (prompt_list, answer_list)
        """
        pass


class BelebelePromptBuilder(PromptBuilder):
    """Prompt builder for Belebele reading comprehension dataset"""

    def build_prompts(self, dataset, split: str, mc_val: List[str]) -> Tuple[List[str], List[int]]:
        prefix_text = "Based on the below passage and the question, choose the best option. \n\n"

        passage = [dataset[split][i]['flores_passage'] for i in range(len(dataset[split]))]
        question = [dataset[split][i]['question'] for i in range(len(dataset[split]))]
        choice1 = [dataset[split][i]['mc_answer1'] for i in range(len(dataset[split]))]
        choice2 = [dataset[split][i]['mc_answer2'] for i in range(len(dataset[split]))]
        choice3 = [dataset[split][i]['mc_answer3'] for i in range(len(dataset[split]))]
        choice4 = [dataset[split][i]['mc_answer4'] for i in range(len(dataset[split]))]
        answer = [dataset[split][i]['correct_answer_num'] for i in range(len(dataset[split]))]

        prompt = [
            prefix_text + 'P: ' + passage[i] + "\nQ: " + question[i] + "\n" +
            mc_val[0] + choice1[i] + "\n" +
            mc_val[1] + choice2[i] + "\n" +
            mc_val[2] + choice3[i] + "\n" +
            mc_val[3] + choice4[i] + "\n" +
            "\nAnswer: "
            for i in range(len(dataset[split]))
        ]

        return prompt, answer


class XCOPAPromptBuilder(PromptBuilder):
    """Prompt builder for XCOPA causal reasoning dataset"""

    def build_prompts(self, dataset, split: str, mc_val: List[str]) -> Tuple[List[str], List[int]]:
        premise = [dataset[split][i]['premise'] for i in range(len(dataset[split]))]
        choice1 = [dataset[split][i]['choice1'] for i in range(len(dataset[split]))]
        choice2 = [dataset[split][i]['choice2'] for i in range(len(dataset[split]))]
        question = [dataset[split][i]['question'] for i in range(len(dataset[split]))]
        answer = [dataset[split][i]['label'] for i in range(len(dataset[split]))]

        prompt = []
        for i in range(len(dataset[split])):
            ce_label = question[i]
            prefix_text = f"Based on the below premise, and two options that reflect the potential {ce_label} to the premise, choose the correct answer. \n\n"
            prompt.append(
                prefix_text + 'Premise: ' + premise[i] + " \n\n\n" +
                mc_val[0] + choice1[i] + "\n" +
                mc_val[1] + choice2[i] + "\n" +
                "\nAnswer: "
            )

        return prompt, answer


class XStoryClozePromptBuilder(PromptBuilder):
    """Prompt builder for XStoryCloze story completion dataset"""

    def build_prompts(self, dataset, split: str, mc_val: List[str]) -> Tuple[List[str], List[int]]:
        prefix_text = "Based on the below story, and two options that reflect potential endings to the story, choose the correct answer. \n\n"

        story1 = [dataset[split][i]['input_sentence_1'] for i in range(len(dataset[split]))]
        story2 = [dataset[split][i]['input_sentence_2'] for i in range(len(dataset[split]))]
        story3 = [dataset[split][i]['input_sentence_3'] for i in range(len(dataset[split]))]
        story4 = [dataset[split][i]['input_sentence_4'] for i in range(len(dataset[split]))]
        choice1 = [dataset[split][i]['sentence_quiz1'] for i in range(len(dataset[split]))]
        choice2 = [dataset[split][i]['sentence_quiz2'] for i in range(len(dataset[split]))]
        answer = [dataset[split][i]['answer_right_ending'] for i in range(len(dataset[split]))]

        prompt = [
            prefix_text + 'Story: ' + story1[i] + " " + story2[i] + " " +
            story3[i] + " " + story4[i] + " " + "\n\n\n" +
            mc_val[0] + choice1[i] + "\n" +
            mc_val[1] + choice2[i] + "\n" +
            "\nAnswer: "
            for i in range(len(dataset[split]))
        ]

        return prompt, answer


# Mapping of dataset names to prompt builders
PROMPT_BUILDERS = {
    'belebele': BelebelePromptBuilder(),
    'xcopa': XCOPAPromptBuilder(),
    'xstorycloze': XStoryClozePromptBuilder()
}


# ============================================================================
# Main Evaluator
# ============================================================================

class MCQAEvaluator:
    """Unified evaluator for multiple-choice QA benchmarks"""

    # Model mappings
    LLM_DICT = {
        'Llama3.1': "meta-llama/Meta-Llama-3.1-8B",
        'Llama3.1it': 'meta-llama/Llama-3.1-8B-Instruct',
        'Aya23': 'CohereForAI/aya-23-8B',
        'Qwen2.5': 'Qwen/Qwen2.5-1.5B'
    }

    def __init__(self, llm_name: str, dataset_name: str):
        """
        Initialize evaluator

        Args:
            llm_name: Name of the LLM (e.g., 'Llama3.1')
            dataset_name: Name of the dataset (e.g., 'belebele')
        """
        self.llm_name = llm_name
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.prompt_builder = PROMPT_BUILDERS[dataset_name]
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the language model and tokenizer"""
        model_path = self.LLM_DICT[self.llm_name]

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("Tokenizer loaded")

        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded")

    def get_target_tokens(self, mc_val: List[str]) -> List[int]:
        """
        Extract token IDs for multiple choice options

        Args:
            mc_val: Multiple choice format (e.g., ['A: ', 'B: '])

        Returns:
            List of token IDs
        """
        # Extract the letter/number from each choice marker
        tokens = [choice.strip().rstrip(':').rstrip('.') for choice in mc_val]
        token_ids = [self.tokenizer.encode(token)[-1] for token in tokens]
        return token_ids

    def run_evaluation(self, lang: str, mc_key: str, batch_size: int = 16) -> List[Dict]:
        """
        Run evaluation on the specified dataset and language

        Args:
            lang: Language code (e.g., 'eng_Latn', 'zh')
            mc_key: Multiple choice format key (e.g., 'ABCD', 'AB')
            batch_size: Batch size for inference

        Returns:
            List of evaluation results
        """
        # Load dataset
        print(f"Loading {self.dataset_name} dataset for language: {lang}")
        dataset = load_dataset(self.config.hf_path, lang)
        print("Dataset loaded")

        # Get multiple choice format
        mc_val = MC_FORMAT_CONFIGS[self.config.num_choices][mc_key]

        # Build prompts
        print("Building prompts...")
        prompt_list, answer_list = self.prompt_builder.build_prompts(
            dataset, self.config.split, mc_val
        )
        print(f"Built {len(prompt_list)} prompts")
        print(f"Example prompt:\n{prompt_list[0]}")
        print(f"Example answer: {answer_list[0]}")

        # Get target tokens
        target_tokens = self.get_target_tokens(mc_val)
        print(f"Target tokens: {target_tokens}")

        # Run batched inference
        evaluation_results = []
        for batch_start in range(0, len(prompt_list), batch_size):
            print(f'Processing batch starting at: {batch_start}')

            batch_prompts = prompt_list[batch_start:batch_start + batch_size]
            batch_answers = answer_list[batch_start:batch_start + batch_size]

            # Tokenize
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.model.device)

            # Get sequence lengths
            lengths = enc["attention_mask"].sum(dim=1)

            # Run inference
            with torch.inference_mode():
                out = self.model(**enc)
                logits = out['logits']

                # Extract last token logits
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                last_indices = lengths - 1
                last_token_logits = logits[batch_indices, last_indices, :]

                # Select only the target token logits
                selected_logits = last_token_logits[:, target_tokens]

            # Compute accuracy
            for i_in_batch, (logit_row, correct_ans) in enumerate(zip(selected_logits, batch_answers)):
                pred_idx = torch.argmax(logit_row).item()
                correct_idx = int(correct_ans) + self.config.answer_offset

                acc = 1.0 if pred_idx == correct_idx else 0.0

                evaluation_results.append({
                    'idx': batch_start + i_in_batch,
                    'acc': acc,
                    'selected_logits': logit_row.tolist(),
                    'correct_ans': correct_idx
                })

        return evaluation_results

    def save_results(self, results: List[Dict], save_dir: str, lang: str, mc_key: str):
        """
        Save evaluation results to JSONL file

        Args:
            results: List of evaluation results
            save_dir: Base directory for saving
            lang: Language code
            mc_key: Multiple choice format key
        """
        # Create save directory
        full_save_dir = os.path.join(save_dir, self.llm_name, self.dataset_name)
        os.makedirs(full_save_dir, exist_ok=True)

        # Create save path
        filename = f"{self.dataset_name}_eval_{self.llm_name}_{lang}_{mc_key}.jsonl"
        save_path = os.path.join(full_save_dir, filename)

        # Write results
        with open(save_path, 'w') as f:
            for record in results:
                f.write(json.dumps(record) + '\n')

        print(f"Results saved to: {save_path}")

        # Print summary statistics
        total = len(results)
        correct = sum(r['acc'] for r in results)
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({int(correct)}/{total})")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified MCQA Benchmark Evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Belebele in English with Llama 3.1
  python mcqa_evaluator.py --dataset belebele --lang eng_Latn --llm_name Llama3.1 --mc_key ABCD

  # Evaluate XCOPA in Chinese with Aya-23
  python mcqa_evaluator.py --dataset xcopa --lang zh --llm_name Aya23 --mc_key AB

  # Evaluate XStoryCloze in Russian with custom batch size
  python mcqa_evaluator.py --dataset xstorycloze --lang ru --llm_name Llama3.1 --mc_key AB --batch_size 32
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['belebele', 'xcopa', 'xstorycloze'],
        help='Dataset to evaluate'
    )
    parser.add_argument(
        '--llm_name',
        type=str,
        default='Llama3.1',
        choices=['Llama3.1', 'Llama3.1it', 'Aya23', 'Qwen2.5'],
        help='Name of the LLM to use'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help='Language code (e.g., eng_Latn for Belebele, zh for XCOPA)'
    )
    parser.add_argument(
        '--mc_key',
        type=str,
        required=True,
        help='Multiple choice format (ABCD/1234/QZRX for 4-choice, AB/12/QZ for 2-choice)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Directory to save results'
    )

    args = parser.parse_args()

    # Validate MC format matches dataset
    config = DATASET_CONFIGS[args.dataset]
    if args.mc_key not in MC_FORMAT_CONFIGS[config.num_choices]:
        valid_keys = list(MC_FORMAT_CONFIGS[config.num_choices].keys())
        raise ValueError(
            f"Invalid mc_key '{args.mc_key}' for {args.dataset}. "
            f"Valid options: {valid_keys}"
        )

    # Initialize evaluator
    print(f"Initializing evaluator for {args.dataset}...")
    evaluator = MCQAEvaluator(args.llm_name, args.dataset)

    # Load model
    evaluator.load_model()

    # Run evaluation
    print(f"\nStarting evaluation...")
    results = evaluator.run_evaluation(args.lang, args.mc_key, args.batch_size)

    # Save results
    evaluator.save_results(results, args.save_dir, args.lang, args.mc_key)


if __name__ == "__main__":
    main()

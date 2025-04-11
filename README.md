# Effect of Cross Lingual Alignment On Multilingual Performance

This repository is the official implementation of "Can you map it to English? The Role of Cross-Lingual Alignment in Multilingual Performance of LLMs", which was submitted to COLM 2025 and can be found here.

## Overview
The folder structure and their corresponding description is provided below. 

```
XLingAlignment/
├── README.md
├── src/
├── data/
├── accuracy_outputs/
├── alignment_outputs/
├── translation_outputs/
```
- **src/**: Contains the code to extract embeddings, compute alignment metrics, and generate translations.
- **data/**: Contains the necessary input files for embeddings and alignment computations.
- **accuracy_outputs/**: Stores accuracy results for various multilingual tasks. Accuracy outputs are based on the default settings of the `lm-harness` tool. 
- **alignment_outputs/**: Contains DALI files (as `.json`) by layer for each sample.
- **translation_outputs/**: Stores translations generated for multilingual tasks.

### src/
- **LLM_generation.py**: Generates the translations from  LLM-s for the corresponding language and translation direction/Evaluates the translation quality for a corresponding language and translation direction. 

### src/alignment/
- **embed_extractor.py**: Extracts embeddings based on the last token and weighted embedding for each sample in a file. Outputs the embeddings as `.pkl` files.
- **compute_dali_xstorycloze.py**: Computes the DALI metric for each xStoryCloze sample using the embeddings file.
- **compute_dali_xcopa.py**: Computes the DALI metric for each XCOPA sample using the embeddings file.
- **compute_dali_belebele.py**: Computes the DALI metric for each Belebele sample using the embeddings file.
- **compute_dalistrong_xstorycloze.py**: Computes the DALI$_S$metric for each xStoryCloze sample using the embeddings file.
- **compute_dalistrong_xcopa.py**: Computes the DALI$_S$ metric for each XCOPA sample using the embeddings file.
- **compute_dalistrong_belebele.py**: Computes the DALI$_S$ metric for each Belebele sample using the embeddings file.
- **compute_mexa.py**: Computes the MEXA metric based on the embedding file. 

### data/
Contains input `.txt` files and datasets for embedding extraction and alignment computation.
- **belebele_dali/**: `.txt` files containing 900 premises (+4 options) for 82 languages.
- **belebele_mexa/**: `.txt` files containing 900 premises for 82 languages.
- **flores_200_dataset/devtest/**: ~ `.txt` files containing 1012 sentences per language from the FLORES dataset.
- **xcopa_dali/**: `/.txt` files containing 500 premises (+2 options) for 10 languages, including English translations.
- **xcopa_mexa/**: `/.txt` files containing 500 premises for 10 languages, including English translations.
- **xstorycloze_dali/**: `/.txt` files containing 1511 stories (+2 options) for 10 languages, plus English.
- **xstorycloze_mexa/**: `/.txt` files containing 1511 stories for 10 languages, plus English.

### accuracy_outputs/
- Contains accuracy results for multilingual tasks, structured by task and language. Accuracy outputs are based on the `lm-harness` tool. 

### alignment_outputs/
- Contains the alignment outputs such as DALI, DALIStrong, and MEXA (as `.json`) by layer for each sample.
- Each DALI/MEXA file is structured as `{Layer_id: {sample_id: 1/0}}`.

### translation_outputs/
- Contains translations generated for multilingual tasks, organized by language, task, and input field. For example, `translation_outputs/Llama3.1/belebele/flores_passage/afrikaans--Llama3.1--english.json` refers to the Afrikaans->English translation of the flores_passage input field of the belebele benchmark.  
- In addition to the translations, the folder also has the COMET scores for each translation. 

## Compute Alignment

### Step 1. Extract Embeddings

In order to extract the embeddings across the layers of the transformer, use the following command:

```python src/alignment/embed_extractor.py -- model_name <MODEL_NAME> --num_sents <NUM_SENTS> --save_path <SAVE_PATH> --data_path <DATA_PATH> --file_ext <FILE_EXT> ```

For example, if we want to extract the embeddings used to calculate the DALI metric for Belebele using Llama3.1 8b, use the following command: 

```python src/alignment/embed_extractor.py -- model_name 'meta-llama/Llama-3.1-8B' --num_sents 900 --save_path <SAVE_PATH> --data_path 'data/belebele_dali/' --file_ext '.txt' ```

Any model hosted on huggingface can be used to compute embeddings. 

### Step 2. Calculate Alignment Score

In order to compute the DALI scores (DALI, DALI$_S$), use the following command:

```python src/alignment/compute_<ALIGNMENT>_<BENCHMARK>.py --llm_name <LLM_NAME> --lang <LANG> --embedding_path <EMBEDDING_PATH> --save_path <SAVE_PATH> ```

For example, if we want to extract DALI for the Belebele benchmark in the `arb_Arab` language using Llama3.1 8b, use the following command: 

```python src/alignment/compute_dali_belebele.py --llm_name 'meta-llama/Llama-3.1-8B'  --lang 'arb_Arab' --embedding_path <EMBEDDING_PATH> --save_path <SAVE_PATH> ```

MEXA, on the other hand is computed using the `compute_mexa.py` command. For example, if we want to extract the MEXA for the Belebele benchmark in the `arb_Arab` language using Llama3.1 8b, use the following command:

```python src/alignment/compute_mexa.py --num_sents 900  --lang 'arb_Arab' --embedding_path <EMBEDDING_PATH> --save_path <SAVE_PATH> --dataset 'belebele'```

Both DALI and MEXA scores are stored as a .json file for each transformer layer per sample. 

## Compute Task Accuracy

Task accuracy is computed using the `lm-harness` tool for the discriminative tasks under consideration: XStorycloze, XCOPA, and Belebele. The default settings of the tool is used, except for the `num_fewshot` parameter which is set to 5 in the case of non-instruction tuned models like Llama3.1. For example, the task accuracy of `arb_Arab` language in the Belebele benchmark using `Llama3.1` model is calculated using the following command: 

```lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks belebele_arb_Arab --num_fewshot 5 --device cuda:0 --batch_size 8 --log_samples --output_path <OUTPUT_PATH>```

The accuracy results for the benchmark are stored in the `accuracy_outputs/` folder for each model. 

## Generate and Compute Translation Quality

To generate the translations using LLM and assess the quality of translations, use the following command:

``` python src/LLM_generation.py --llm_name <LLM_NAME> --source_lang <SOURCE_LANG> --target_lang <TARGET_LANG> --translation_task <TRANSLATION_TASK> --save_dir <SAVE_DIR> --strata <STRATA> --num_fewshot <NUM_FEWSHOT> --dataset <DATASET> --combination_approach <COMBINATION_APPROACH> --input_field <INPUT_FIELD> --translation_mode <TRANSLATION_MODE> ```

For example, if we want to translate the `arb_Arab` passages in the Belebele benchmark to `eng_Latn` using Llama3.1, we use:

``` python src/LLM_generation.py --llm_name 'meta-llama/Llama-3.1-8B' --source_lang 'arb_Arab' --translation_task 'xxtoen' --save_dir <SAVE_DIR> --strata 'test' --num_fewshot 5 --dataset 'belebele' --combination_approach 'separate_inputs' --input_field 'flores_passage' --translation_mode 'translation' ```

Once the translations are generated, and we want to evaluate the quality of translations, we run the same command but with the `translation_mode` argument set to `evaluation` instead. All the translations for Llama3.1 model are already run and stored in `translation_outpus/` directory. 











